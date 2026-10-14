/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <doca_gpunetio_dev_verbs_twosided.cuh>

#include <cuda/atomic>

#include "src/engines/gpunetio_rdma/daqiri_gpunetio_rdma_kernels.h"

namespace daqiri::gpunetio_rdma {

namespace {

constexpr uint32_t THREADS = 128;
constexpr uint32_t DESC_MASK = DESC_RING - 1;
// Pending send-queue operations: send bursts plus one memory consistency WQE per receive burst
constexpr uint32_t SQ_FIFO = 2 * DESC_RING;
constexpr uint32_t IDLE_SLEEP_NS = 2000;

static_assert((DESC_RING & DESC_MASK) == 0, "DESC_RING must be a power of two");

// Only thread 0 of a block touches the queue pair state, so the DOCA helpers run in exclusive mode
constexpr auto EXCLUSIVE = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;

// Flags, counters and rings in pinned host memory are shared with the CPU
template <typename T>
__device__ __forceinline__ T load_acquire_sys(const T* ptr) {
  cuda::atomic_ref<T, cuda::thread_scope_system> ref(*const_cast<T*>(ptr));
  return ref.load(cuda::std::memory_order_acquire);
}

// Relaxed system-scope loads bypass the L1 cache, which may hold an older copy of a reused entry
template <typename T>
__device__ __forceinline__ T load_relaxed_sys(const T* ptr) {
  cuda::atomic_ref<T, cuda::thread_scope_system> ref(*const_cast<T*>(ptr));
  return ref.load(cuda::std::memory_order_relaxed);
}

template <typename T>
__device__ __forceinline__ void store_release_sys(T* ptr, T value) {
  cuda::atomic_ref<T, cuda::thread_scope_system> ref(*ptr);
  ref.store(value, cuda::std::memory_order_release);
}

template <typename T>
__device__ __forceinline__ void store_relaxed_sys(T* ptr, T value) {
  cuda::atomic_ref<T, cuda::thread_scope_system> ref(*ptr);
  ref.store(value, cuda::std::memory_order_relaxed);
}

enum Command : uint32_t {
  CMD_IDLE = 0,
  CMD_POLL = 1,
  CMD_POST_SEND = 2,
  CMD_POST_RECV = 3,
  CMD_EXIT = 4,
};

// A send-queue operation waiting for its completion
enum SqKind : uint32_t {
  SQ_SEND_BURST = 0,
  SQ_DUMP = 1,  // memory consistency WQE of a receive burst
};

struct SqEntry {
  uint64_t last_ticket;  // ticket of the last WQE of the operation
  uint32_t kind;         // SqKind
  uint32_t burst;        // send burst, or receive burst of a dump, modulo DESC_RING
};

// Progress of a receive burst
enum RqProgress : uint32_t {
  RQ_POSTED = 0,
  RQ_DUMPING = 1,
  RQ_DONE = 2,
};

// Block state. Thread 0 owns it; the other threads read the posting fields after a barrier.
struct BlockState {
  uint32_t state;  // SlotState last seen
  uint32_t cmd;
  bool prefer_send;

  // Burst being posted
  uint64_t base;  // first ticket
  uint32_t first_pkt;
  uint32_t num_pkts;
  uint32_t flags;

  // Send side: bursts are read, posted and completed in order
  uint64_t sq_posted;
  uint64_t sq_done;
  uint64_t sq_head;
  uint64_t sq_tail;
  SqEntry sq_fifo[SQ_FIFO];

  // Receive side: [rq_done, rq_polled) are polled bursts waiting for a dump or for an earlier
  // burst, [rq_polled, rq_posted) are posted bursts
  uint64_t rq_posted;
  uint64_t rq_polled;
  uint64_t rq_done;
  uint64_t rq_next_ticket;  // next receive completion to poll
  uint64_t rq_first_ticket[DESC_RING];
  uint64_t rq_last_ticket[DESC_RING];
  uint32_t rq_first_pkt[DESC_RING];
  uint32_t rq_flags[DESC_RING];
  uint32_t rq_status[DESC_RING];
  uint32_t rq_progress[DESC_RING];
};

__device__ void reset_block_state(BlockState& st) {
  st.sq_posted = 0;
  st.sq_done = 0;
  st.sq_head = 0;
  st.sq_tail = 0;
  st.rq_posted = 0;
  st.rq_polled = 0;
  st.rq_done = 0;
  st.rq_next_ticket = 0;
  st.prefer_send = false;
}

__device__ bool desc_ready(const BurstDesc* ring, uint64_t index) {
  return load_acquire_sys(&ring[index & DESC_MASK].ready) == index + 1;
}

// Records the first error completion that isn't a flush, for the host log
__device__ void record_error(const SlotArgs& args, uint32_t direction,
                             struct doca_gpu_dev_verbs_cqe64* cqe) {
  if (cqe == nullptr) {
    return;
  }
  const auto* err = reinterpret_cast<const struct mlx5_err_cqe*>(cqe);
  const uint32_t syndrome = err->syndrome;
  if (syndrome == MLX5_CQE_SYNDROME_WR_FLUSH_ERR || load_relaxed_sys(&args.ctrl->error) != 0) {
    return;
  }
  store_relaxed_sys(&args.ctrl->error, (direction << 16) | (syndrome << 8) |
                                           static_cast<uint32_t>(err->vendor_err_synd));
}

// The DOCA poll functions count a completion as consumed only when it succeeded. Error completions
// are consumed too, or the poll window would stop moving once the queue pair is in error.
__device__ void consume_error(struct doca_gpu_dev_verbs_cq* cq, uint64_t ticket) {
  if (cq->cqe_ci < ticket + 1) {
    cq->cqe_ci = ticket + 1;
  }
}

// Rings the doorbell for [base, base + count). The WQEs written by the other threads are visible
// to this thread after the barrier; the release fence orders them before the doorbell for the NIC.
template <enum doca_gpu_dev_verbs_qp_type qp_type>
__device__ void submit(struct doca_gpu_dev_verbs_qp* qp, uint64_t base, uint32_t count) {
  doca_gpu_dev_verbs_mark_wqes_ready<EXCLUSIVE, qp_type>(qp, base, base + count - 1);
  doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
  doca_gpu_dev_verbs_submit<EXCLUSIVE, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                            DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO, qp_type>(qp, base + count);
}

// Thread 0: picks the next action of the block and reserves the WQEs of a burst to post
__device__ void plan(const SlotArgs& args, BlockState& st) {
  if (load_acquire_sys(&args.ctrl->exit) != 0) {
    st.cmd = CMD_EXIT;
    return;
  }
  const uint32_t state = load_acquire_sys(&args.ctrl->state);
  if (state == SLOT_ACTIVE && st.state != SLOT_ACTIVE) {
    // A new connection: the host reset the rings and the counters
    reset_block_state(st);
  }
  st.state = state;
  if (state == SLOT_CLOSING) {
    st.cmd = CMD_POLL;
    return;
  }
  if (state != SLOT_ACTIVE) {
    st.cmd = CMD_IDLE;
    return;
  }

  // Alternate between the directions so that neither starves the other
  const bool send_ready = desc_ready(args.sq_desc, st.sq_posted);
  const bool recv_ready = desc_ready(args.rq_desc, st.rq_posted);
  const bool send = send_ready && (!recv_ready || st.prefer_send);
  if (!send && !recv_ready) {
    st.cmd = CMD_POLL;
    return;
  }
  st.prefer_send = !send;
  const BurstDesc* desc =
      send ? &args.sq_desc[st.sq_posted & DESC_MASK] : &args.rq_desc[st.rq_posted & DESC_MASK];
  st.first_pkt = load_relaxed_sys(&desc->first_pkt);
  st.num_pkts = load_relaxed_sys(&desc->num_pkts);
  st.flags = load_relaxed_sys(&desc->flags);
  // The host never publishes more WQEs than the queue holds, so the slots are free already
  if (send) {
    st.base = doca_gpu_dev_verbs_reserve_wq_slots<EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_SQ>(
        args.qp, st.num_pkts, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
    st.cmd = CMD_POST_SEND;
  } else {
    st.base = doca_gpu_dev_verbs_reserve_wq_slots<EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_RQ>(
        args.qp, st.num_pkts, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
    st.cmd = CMD_POST_RECV;
  }
}

// All threads: write the WQEs of the burst being posted, one per message. Every WQE asks for a
// completion, so the completion of ticket t is entry t of the completion queue.
__device__ void write_wqes(const SlotArgs& args, const BlockState& st, bool send) {
  for (uint32_t i = threadIdx.x; i < st.num_pkts; i += blockDim.x) {
    const uint64_t ticket = st.base + i;
    const Packet* pkt = send ? &args.sq_pkts[(st.first_pkt + i) & args.sq_pkt_mask]
                             : &args.rq_pkts[(st.first_pkt + i) & args.rq_pkt_mask];
    const uint64_t addr = load_relaxed_sys(&pkt->addr);
    const uint32_t len = load_relaxed_sys(&pkt->len);
    const uint32_t lkey = load_relaxed_sys(&pkt->lkey);
    if (send) {
      struct doca_gpu_dev_verbs_wqe* wqe =
          doca_gpu_dev_verbs_get_wqe_ptr(args.qp, static_cast<uint16_t>(ticket));
      doca_gpu_dev_verbs_wqe_prepare_send(
          args.qp, wqe, static_cast<uint16_t>(ticket), DOCA_GPUNETIO_MLX5_OPCODE_SEND,
          DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, 0, addr, lkey, len);
    } else {
      struct mlx5_wqe_data_seg* rwqe =
          doca_gpu_dev_verbs_get_rwqe_ptr(args.qp, static_cast<uint16_t>(ticket));
      doca_gpu_dev_verbs_wqe_prepare_recv(args.qp, rwqe, addr, lkey, len);
    }
  }
}

// Thread 0: hands the WQEs written by the block to the NIC and tracks their completion
__device__ void commit(const SlotArgs& args, BlockState& st, bool send) {
  if (send) {
    submit<DOCA_GPUNETIO_VERBS_QP_SQ>(args.qp, st.base, st.num_pkts);
    SqEntry& entry = st.sq_fifo[st.sq_tail % SQ_FIFO];
    entry.last_ticket = st.base + st.num_pkts - 1;
    entry.kind = SQ_SEND_BURST;
    entry.burst = static_cast<uint32_t>(st.sq_posted & DESC_MASK);
    st.sq_tail++;
    st.sq_posted++;
  } else {
    submit<DOCA_GPUNETIO_VERBS_QP_RQ>(args.qp, st.base, st.num_pkts);
    const uint32_t idx = static_cast<uint32_t>(st.rq_posted & DESC_MASK);
    st.rq_first_ticket[idx] = st.base;
    st.rq_last_ticket[idx] = st.base + st.num_pkts - 1;
    st.rq_first_pkt[idx] = st.first_pkt;
    st.rq_flags[idx] = st.flags;
    st.rq_status[idx] = BURST_OK;
    st.rq_progress[idx] = RQ_POSTED;
    st.rq_posted++;
  }
}

// Thread 0: posts the memory consistency WQE of a receive burst. Pre-Hopper GPUs may not see the
// data the NIC wrote into GPU memory until a later NIC operation completes.
__device__ void post_dump(const SlotArgs& args, BlockState& st, uint32_t idx) {
  const uint64_t ticket = doca_gpu_dev_verbs_reserve_wq_slots<EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_SQ>(
      args.qp, 1, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
  struct doca_gpu_dev_verbs_wqe* wqe =
      doca_gpu_dev_verbs_get_wqe_ptr(args.qp, static_cast<uint16_t>(ticket));
  doca_gpu_dev_verbs_wqe_prepare_dump(args.qp, wqe, static_cast<uint16_t>(ticket),
                                      DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, args.dump_addr,
                                      args.dump_lkey, 1);
  submit<DOCA_GPUNETIO_VERBS_QP_SQ>(args.qp, ticket, 1);
  SqEntry& entry = st.sq_fifo[st.sq_tail % SQ_FIFO];
  entry.last_ticket = ticket;
  entry.kind = SQ_DUMP;
  entry.burst = idx;
  st.sq_tail++;
  st.rq_progress[idx] = RQ_DUMPING;
}

// Thread 0: send completions arrive in order; a burst is complete with its last WQE
__device__ void poll_send(const SlotArgs& args, BlockState& st) {
  while (st.sq_head != st.sq_tail) {
    const SqEntry& entry = st.sq_fifo[st.sq_head % SQ_FIFO];
    struct doca_gpu_dev_verbs_cqe64* cqe = nullptr;
    const int ret = doca_gpu_dev_verbs_poll_one_cq_at<EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_SQ>(
        args.qp, entry.last_ticket, &cqe);
    if (ret == EBUSY) {
      return;
    }
    const uint32_t status = ret == 0 ? BURST_OK : BURST_ERROR;
    if (ret != 0) {
      record_error(args, ERROR_SEND, cqe);
      consume_error(&args.qp->cq_sq, entry.last_ticket);
    }
    if (entry.kind == SQ_SEND_BURST) {
      store_relaxed_sys(&args.sq_desc[entry.burst].status, status);
      st.sq_done++;
      store_release_sys(&args.ctrl->sq_done, st.sq_done);
    } else {
      if (status != BURST_OK) {
        st.rq_status[entry.burst] = BURST_ERROR;
      }
      st.rq_progress[entry.burst] = RQ_DONE;
    }
    st.sq_head++;
  }
}

// Thread 0: receive completions arrive in posting order. The received byte counts go back to the
// host through the packet ring; bursts are reported in order once complete.
__device__ void poll_recv(const SlotArgs& args, BlockState& st) {
  const bool need_dump = args.qp->need_mcst && st.state == SLOT_ACTIVE;
  while (st.rq_polled < st.rq_posted) {
    const uint32_t idx = static_cast<uint32_t>(st.rq_polled & DESC_MASK);
    while (st.rq_next_ticket <= st.rq_last_ticket[idx]) {
      struct doca_gpu_dev_verbs_cqe64* cqe = nullptr;
      const int ret = doca_gpu_dev_verbs_poll_one_cq_at<EXCLUSIVE, DOCA_GPUNETIO_VERBS_QP_RQ>(
          args.qp, st.rq_next_ticket, &cqe);
      if (ret == EBUSY) {
        goto report;
      }
      const uint32_t pkt = (st.rq_first_pkt[idx] +
                            static_cast<uint32_t>(st.rq_next_ticket - st.rq_first_ticket[idx])) &
                           args.rq_pkt_mask;
      if (ret == 0) {
        store_relaxed_sys(&args.rq_pkts[pkt].len, doca_gpu_dev_verbs_cqe_get_bytes(cqe));
      } else {
        record_error(args, ERROR_RECV, cqe);
        consume_error(&args.qp->cq_rq, st.rq_next_ticket);
        st.rq_status[idx] = BURST_ERROR;
        store_relaxed_sys(&args.rq_pkts[pkt].len, 0u);
      }
      st.rq_next_ticket++;
    }
    if (need_dump && (st.rq_flags[idx] & DESC_FLAG_GPU_BUFFERS) != 0 &&
        st.rq_status[idx] == BURST_OK) {
      post_dump(args, st, idx);
    } else {
      st.rq_progress[idx] = RQ_DONE;
    }
    st.rq_polled++;
  }

report:
  while (st.rq_done < st.rq_polled && st.rq_progress[st.rq_done & DESC_MASK] == RQ_DONE) {
    const uint32_t idx = static_cast<uint32_t>(st.rq_done & DESC_MASK);
    store_relaxed_sys(&args.rq_desc[idx].status, st.rq_status[idx]);
    st.rq_done++;
    store_release_sys(&args.ctrl->rq_done, st.rq_done);
  }
}

// Thread 0, closing slot: the host moved the queue pair to the error state, so every posted WQE
// completes, with a flush error if needed. Bursts published but not posted yet complete as flushed.
// Once nothing is pending, the GPU side of the queue pair is reset for the next connection, which
// starts from ticket 0 on the same completion queues.
__device__ void drain(const SlotArgs& args, BlockState& st) {
  if (st.sq_head == st.sq_tail) {
    while (desc_ready(args.sq_desc, st.sq_posted)) {
      store_relaxed_sys(&args.sq_desc[st.sq_posted & DESC_MASK].status,
                        static_cast<uint32_t>(BURST_FLUSHED));
      st.sq_posted++;
      st.sq_done++;
      store_release_sys(&args.ctrl->sq_done, st.sq_done);
    }
  }
  if (st.rq_done == st.rq_posted) {
    while (desc_ready(args.rq_desc, st.rq_posted)) {
      store_relaxed_sys(&args.rq_desc[st.rq_posted & DESC_MASK].status,
                        static_cast<uint32_t>(BURST_FLUSHED));
      st.rq_posted++;
      st.rq_polled++;
      st.rq_done++;
      store_release_sys(&args.ctrl->rq_done, st.rq_done);
    }
  }
  if (st.sq_head != st.sq_tail || st.rq_done != st.rq_posted) {
    return;
  }

  struct doca_gpu_dev_verbs_qp* qp = args.qp;
  // Every posted WQE produced exactly one completion: skip them for the next connection
  qp->cq_sq.cqe_rsvd += qp->sq_wqe_pi;
  qp->cq_sq.cqe_ci = 0;
  qp->cq_rq.cqe_rsvd += qp->rq_wqe_pi;
  qp->cq_rq.cqe_ci = 0;
  qp->sq_rsvd_index = 0;
  qp->sq_ready_index = 0;
  qp->sq_wqe_pi = 0;
  qp->rq_rsvd_index = 0;
  qp->rq_ready_index = 0;
  qp->rq_wqe_pi = 0;
  qp->last_get_wqe_idx = 0;
  qp->last_dump_wqe_idx = 0;
  qp->last_recv_wqe_idx = 0;
  // The queue pair restarts from producer index 0 after the reset
  store_relaxed_sys(qp->sq_dbrec, static_cast<__be32>(0));
  store_relaxed_sys(qp->rq_dbrec, static_cast<__be32>(0));
  __threadfence_system();
  reset_block_state(st);
  st.state = SLOT_DRAINED;
  store_release_sys(&args.ctrl->state, static_cast<uint32_t>(SLOT_DRAINED));
}

__global__ void __launch_bounds__(THREADS) verbs_kernel(const SlotArgs* slots) {
  __shared__ BlockState st;
  const SlotArgs args = slots[blockIdx.x];

  if (threadIdx.x == 0) {
    reset_block_state(st);
    st.state = SLOT_IDLE;
    st.cmd = CMD_IDLE;
  }
  __syncthreads();

  while (true) {
    if (threadIdx.x == 0) {
      plan(args, st);
    }
    __syncthreads();
    const uint32_t cmd = st.cmd;
    if (cmd == CMD_EXIT) {
      return;
    }
    if (cmd == CMD_POST_SEND || cmd == CMD_POST_RECV) {
      write_wqes(args, st, cmd == CMD_POST_SEND);
      __syncthreads();
      if (threadIdx.x == 0) {
        commit(args, st, cmd == CMD_POST_SEND);
      }
    }
    if (threadIdx.x == 0) {
      if (st.state == SLOT_ACTIVE || st.state == SLOT_CLOSING) {
        poll_send(args, st);
        poll_recv(args, st);
        if (st.state == SLOT_CLOSING) {
          drain(args, st);
        }
      }
      if (cmd == CMD_IDLE) {
        __nanosleep(IDLE_SLEEP_NS);
      }
    }
    __syncthreads();
  }
}

}  // namespace

uint32_t kernel_threads() {
  return THREADS;
}

cudaError_t load_kernels() {
  cudaFuncAttributes attr;
  return cudaFuncGetAttributes(&attr, verbs_kernel);
}

cudaError_t check_capacity(uint32_t num_slots) {
  int device = 0;
  int sms = 0;
  int per_sm = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  }
  if (err == cudaSuccess) {
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, verbs_kernel, THREADS, 0);
  }
  if (err != cudaSuccess) {
    return err;
  }
  return static_cast<uint64_t>(per_sm) * sms >= num_slots ? cudaSuccess
                                                          : cudaErrorLaunchOutOfResources;
}

cudaError_t launch_kernel(const SlotArgs* slots, uint32_t num_slots, cudaStream_t stream) {
  const cudaError_t err = check_capacity(num_slots);
  if (err != cudaSuccess) {
    return err;
  }
  verbs_kernel<<<num_slots, THREADS, 0, stream>>>(slots);
  return cudaGetLastError();
}

}  // namespace daqiri::gpunetio_rdma
