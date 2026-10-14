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

#include <doca_gpunetio_dev_verbs_qp.cuh>

#include <cuda/atomic>

#include "src/engines/gpunetio_verbs/daqiri_gpunetio_verbs_kernels.h"

namespace daqiri::gpunetio_verbs {

namespace {

constexpr uint32_t RX_THREADS = 256;
// A polling round of the RX kernel returns at least this often, to apply releases and check for
// exit
constexpr uint64_t RX_POLL_NS = 5000;

constexpr uint32_t TX_THREADS = 256;
// Chunks in flight on a TX queue, each one waits for a completion
constexpr uint32_t TX_CHUNKS_IN_FLIGHT = 64;
// While the TX kernel waits for a completion, it checks for exit this often
constexpr uint32_t TX_EXIT_CHECK_SPINS = 1024;

// The NIC computes the IPv4 and the TCP/UDP checksums of every packet sent
constexpr uint8_t TX_CS_FLAGS = MLX5_ETH_WQE_L3_CSUM | MLX5_ETH_WQE_L4_CSUM;

constexpr auto SHARING = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
constexpr auto SCOPE = DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
constexpr auto HANDLER = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;

// Flags and counters in pinned host memory are shared with the CPU
template <typename T>
__device__ __forceinline__ T load_acquire_sys(T* ptr) {
  cuda::atomic_ref<T, cuda::thread_scope_system> ref(*ptr);
  return ref.load(cuda::std::memory_order_acquire);
}

template <typename T>
__device__ __forceinline__ void store_release_sys(T* ptr, T value) {
  cuda::atomic_ref<T, cuda::thread_scope_system> ref(*ptr);
  ref.store(value, cuda::std::memory_order_release);
}

/*
 * Completion queues. The NIC writes the CQEs in order, one per position, and flips their owner
 * bit at each pass over the queue. The kernels track their positions themselves: the queues are
 * created in overrun mode, so the NIC never waits for the consumer index.
 */

enum class Cqe { BUSY, OK, ERROR };

struct CqView {
  uint8_t* cqes;
  uint32_t num;  // Power of two
};

__device__ __forceinline__ CqView cq_view(doca_gpu_dev_verbs_cq* cq) {
  return CqView{cq->cqe_daddr, cq->cqe_num};
}

__device__ __forceinline__ doca_gpu_dev_verbs_cqe64* cqe_at(const CqView& cq, uint64_t pos) {
  return reinterpret_cast<doca_gpu_dev_verbs_cqe64*>(cq.cqes + (pos & (cq.num - 1)) *
                                                                   DOCA_GPUNETIO_VERBS_CQE_SIZE);
}

// State of the CQE at position pos, which completes the work request wqe_counter. As the DOCA
// GPUNetIO polling functions do, the work request counter is checked only before Hopper, where the
// GPU may see the owner bit before the rest of the CQE.
__device__ __forceinline__ Cqe cqe_state(const CqView& cq, uint64_t pos, uint16_t wqe_counter) {
  doca_gpu_dev_verbs_cqe64* cqe = cqe_at(cq, pos);
  const uint32_t chunk = doca_gpu_dev_verbs_bswap32(
      doca_gpu_dev_verbs_load_relaxed_sys_global(&cqe->wqe_counter_sig_op_own_raw));
  const uint8_t opown = chunk & 0xff;
  const uint8_t opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
  if (opcode == MLX5_CQE_INVALID || (opown & MLX5_CQE_OWNER_MASK) != ((pos & cq.num) != 0)) {
    return Cqe::BUSY;
  }
  if (opcode == MLX5_CQE_REQ_ERR || opcode == MLX5_CQE_RESP_ERR) {
    return Cqe::ERROR;
  }
#if __CUDA_ARCH__ < 900
  if (static_cast<uint16_t>(chunk >> 16) != wqe_counter) {
    return Cqe::BUSY;
  }
#else
  (void)wqe_counter;
#endif
  return Cqe::OK;
}

__device__ __forceinline__ uint32_t cqe_load32(__be32* field) {
  return doca_gpu_dev_verbs_bswap32(
      doca_gpu_dev_verbs_load_relaxed_sys_global(reinterpret_cast<uint32_t*>(field)));
}

__device__ __forceinline__ uint32_t cqe_syndrome(const CqView& cq, uint64_t pos) {
  auto* err = reinterpret_cast<struct mlx5_err_cqe_ex*>(cqe_at(cq, pos));
  return doca_gpu_dev_verbs_load_relaxed_sys_global(&err->syndrome);
}

__device__ __forceinline__ uint16_t cqe_wqe_counter(const CqView& cq, uint64_t pos) {
  return static_cast<uint16_t>(cqe_load32(&cqe_at(cq, pos)->wqe_counter_sig_op_own_raw) >> 16);
}

/*
 * Receive path. Receive work request i of the ring points at ring slot i of the queue's memory
 * region; the engine writes them before the kernel starts and they never change. The kernel keeps
 * the received packets out of the NIC until the application frees them: the CPU reports the freed
 * packets, oldest first, and only then does the kernel post their work requests again.
 */

// Thread 0: posts receive work requests up to producer index pi
__device__ __forceinline__ void rx_post(doca_gpu_dev_verbs_qp* qp, uint64_t pi) {
  doca_gpu_dev_verbs_submit<SHARING, SCOPE, HANDLER, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, pi);
}

/*
 * All threads: counts the completions that arrived from position base, waiting up to timeout_ns
 * for max_pkts of them. Thread i checks positions base + i, base + i + blockDim.x, ... The count
 * covers only positions some thread saw complete, so it never includes a completion still in
 * flight. A completion with error right after the counted ones sets failed.
 */
__device__ uint32_t rx_poll(const CqView& cq, uint64_t base, uint32_t max_pkts, uint64_t timeout_ns,
                            bool* failed) {
  __shared__ uint32_t first_busy;
  __shared__ uint32_t first_error;
  const uint32_t per_thread = max_pkts / blockDim.x;
  const uint64_t t0 = threadIdx.x == 0 ? doca_gpu_dev_verbs_query_globaltimer() : 0;
  uint32_t got = 0;
  bool error = false;

  if (threadIdx.x == 0) {
    first_busy = max_pkts;
    first_error = max_pkts;
  }
  __syncthreads();

  while (true) {
    bool stop = false;
    if (got < per_thread && !error) {
      const uint64_t pos = base + threadIdx.x + static_cast<uint64_t>(got) * blockDim.x;
      const Cqe state = cqe_state(cq, pos, static_cast<uint16_t>(pos));
      if (state == Cqe::OK) {
        stop = ++got == per_thread;
      } else if (state == Cqe::ERROR) {
        error = true;
        stop = true;
      }
    }
    if (threadIdx.x == 0 && doca_gpu_dev_verbs_query_globaltimer() - t0 > timeout_ns) {
      stop = true;
    }
    if (__syncthreads_or(stop)) {
      break;
    }
  }

  const uint32_t mine = threadIdx.x + got * blockDim.x;
  if (got < per_thread) {
    atomicMin_block(&first_busy, mine);
  }
  if (error) {
    atomicMin_block(&first_error, mine);
  }
  __syncthreads();
  // An error completion counts only once every completion before it was seen
  *failed = first_error < max_pkts && first_error == first_busy;
  return first_busy;
}

// All threads: the byte counts and flow tags of the count packets polled from position base, for
// the CPU, at their ring slots
__device__ void rx_store(const RxKernelArgs& args, const CqView& cq, uint64_t base,
                         uint32_t count) {
  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    doca_gpu_dev_verbs_cqe64* cqe = cqe_at(cq, base + i);
    const uint64_t slot = (base + i) & args.ring_mask;
    args.pkt_len[slot] = cqe_load32(&cqe->byte_cnt);
    // A steering rule with a flow ID tags its packets in the low 24 bits
    args.flow_tag[slot] = cqe_load32(&cqe->sop_drop_qpn) & 0x00ffffffU;
  }
  __threadfence_system();
}

/*
 * One block per RX queue of the GPU: block i receives from queues[i] for the lifetime of the
 * engine. A packet stays valid until the application frees its burst.
 */
__global__ void __launch_bounds__(RX_THREADS) rx_kernel(const RxKernelArgs* queues) {
  const RxKernelArgs args = queues[blockIdx.x];
  const CqView cq = cq_view(doca_gpu_dev_verbs_qp_get_cq_rq(args.qp));
  const uint64_t ring_size = static_cast<uint64_t>(args.ring_mask) + 1;
  __shared__ uint64_t received;  // Packets received so far; ring slot = sequence % ring size
  __shared__ uint32_t stop;

  // Used by thread 0 only
  uint64_t released = 0;
  uint64_t burst_start = 0;  // First packet of the burst being collected
  uint64_t burst_pkts = 0;
  uint64_t burst_t0 = 0;
  uint64_t next_desc = 0;

  if (threadIdx.x == 0) {
    received = 0;
    stop = 0;
    // Every ring slot starts with the NIC
    rx_post(args.qp, ring_size);
  }
  __syncthreads();

  while (true) {
    // Thread 0 updates the shared state only once every thread is done with it
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint64_t to_release = load_acquire_sys(&args.ctrl->released_pkts);
      if (to_release != released) {
        released = to_release;
        rx_post(args.qp, ring_size + released);
      }
      if (load_acquire_sys(&args.ctrl->exit) != 0) {
        stop = 1;
      }
    }
    __syncthreads();
    if (stop != 0) {
      break;
    }

    const uint64_t base = received;
    bool failed = false;
    const uint32_t count = rx_poll(cq, base, args.max_pkts, RX_POLL_NS, &failed);
    if (count > 0) {
      rx_store(args, cq, base, count);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      if (count > 0 && cqe_wqe_counter(cq, base) != static_cast<uint16_t>(base)) {
        store_release_sys(&args.ctrl->error, static_cast<uint32_t>(KERNEL_RX_OUT_OF_ORDER));
        stop = 1;
      } else {
        if (count > 0) {
          if (burst_pkts == 0) {
            burst_start = base;
            burst_t0 = doca_gpu_dev_verbs_query_globaltimer();
          }
          burst_pkts += count;
          received = base + count;
        }

        // Publish full bursts. A burst also ends at the end of the ring, so that its packets are
        // contiguous in memory, when the timeout elapses, and before the block stops on an error.
        const uint64_t now =
            (burst_pkts > 0 && args.timeout_ns > 0) ? doca_gpu_dev_verbs_query_globaltimer() : 0;
        const uint64_t published = next_desc;
        while (burst_pkts > 0) {
          const uint64_t room = ring_size - (burst_start & args.ring_mask);
          uint64_t publish;
          if (burst_pkts >= args.batch_size && args.batch_size <= room) {
            publish = args.batch_size;
          } else if (burst_pkts >= room) {
            publish = room;
          } else if (failed || (args.timeout_ns > 0 && now - burst_t0 >= args.timeout_ns)) {
            publish = burst_pkts;
          } else {
            break;
          }
          RxBurstDesc* desc = &args.desc[next_desc & args.desc_mask];
          desc->pkt_seq = burst_start;
          desc->num_pkts = static_cast<uint32_t>(publish);
          store_release_sys(&desc->ready, static_cast<uint32_t>(next_desc + 1));
          next_desc++;
          burst_start += publish;
          burst_pkts -= publish;
          burst_t0 = now;
        }
        if (count > 0 || next_desc != published) {
          store_release_sys(&args.ctrl->pkts, received);
          store_release_sys(&args.ctrl->bursts, next_desc);
        }

        if (failed) {
          args.ctrl->syndrome = cqe_syndrome(cq, base + count);
          store_release_sys(&args.ctrl->error, static_cast<uint32_t>(KERNEL_RX_ERROR_CQE));
          stop = 1;
        }
      }
    }
  }
}

/*
 * Send path. Each packet takes one send work request of one WQEBB: a control segment, an Ethernet
 * segment that asks for the checksums, and a data segment pointing at the packet in its TX slot.
 * The kernel posts each burst as chunks of work requests. Only the last work request of a chunk
 * asks the NIC for a completion, and a chunk is never bigger than half the send queue, so the
 * queue always takes a new chunk once the oldest chunks complete.
 */
struct TxRun {
  uint64_t next_wqe;
  uint64_t done_wqe;
  uint64_t chunks;
  uint64_t cqes;
  uint64_t bursts_done;
  uint64_t chunk_end[TX_CHUNKS_IN_FLIGHT];  // Work request index after the chunk
  uint8_t chunk_last[TX_CHUNKS_IN_FLIGHT];  // The chunk ends a burst
};

// Thread 0: consumes the completion of the oldest chunk in flight. Without wait, returns with
// progress == false if the completion has not arrived yet. Returns false on a completion error,
// and when the engine asks the kernel to exit while it waits.
__device__ bool tx_complete_one(const TxKernelArgs& args, const CqView& cq, TxRun& run, bool wait,
                                bool* progress) {
  *progress = false;
  if (run.cqes == run.chunks) {
    return true;
  }
  const uint32_t slot = run.cqes % TX_CHUNKS_IN_FLIGHT;
  const uint64_t last_wqe = run.chunk_end[slot] - 1;
  Cqe state = cqe_state(cq, run.cqes, static_cast<uint16_t>(last_wqe));
  for (uint32_t spins = 1; wait && state == Cqe::BUSY; spins++) {
    if (spins % TX_EXIT_CHECK_SPINS == 0 && load_acquire_sys(&args.ctrl->exit) != 0) {
      return false;
    }
    state = cqe_state(cq, run.cqes, static_cast<uint16_t>(last_wqe));
  }
  if (state == Cqe::BUSY) {
    return true;
  }
  if (state == Cqe::ERROR) {
    args.ctrl->syndrome = cqe_syndrome(cq, run.cqes);
    store_release_sys(&args.ctrl->error, static_cast<uint32_t>(KERNEL_TX_ERROR_CQE));
    return false;
  }
  if (cqe_wqe_counter(cq, run.cqes) != static_cast<uint16_t>(last_wqe)) {
    store_release_sys(&args.ctrl->error, static_cast<uint32_t>(KERNEL_TX_OUT_OF_ORDER));
    return false;
  }
  run.done_wqe = run.chunk_end[slot];
  if (run.chunk_last[slot] != 0) {
    run.bursts_done++;
    store_release_sys(&args.ctrl->completed_bursts, run.bursts_done);
  }
  run.cqes++;
  *progress = true;
  return true;
}

// All threads: one work request per packet, then thread 0 rings the doorbell
__device__ void tx_post_chunk(const TxKernelArgs& args, uint64_t wqe_first, uint64_t slot_first,
                              uint32_t count) {
  doca_gpu_dev_verbs_qp* qp = args.qp;
  const uint32_t qpn_ds = doca_gpu_dev_verbs_load_const(&qp->sq_num_shift8_be_3ds);
  for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    // The CPU fills the packet table: read it from memory, not from a cache
    const TxPacket* pkt = &args.pkts[(slot_first + i) % args.num_slots];
    const uint64_t addr =
        doca_gpu_dev_verbs_load_relaxed_sys_global(const_cast<uint64_t*>(&pkt->addr));
    const uint64_t len =
        doca_gpu_dev_verbs_load_relaxed_sys_global(const_cast<uint64_t*>(&pkt->len));
    const uint64_t wqe_idx = wqe_first + i;

    // The control segment holds the work request index in 16 bits, below the opmod byte
    struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
    cseg.opmod_idx_opcode =
        doca_gpu_dev_verbs_bswap32((static_cast<uint32_t>(static_cast<uint16_t>(wqe_idx))
                                    << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                   DOCA_GPUNETIO_MLX5_OPCODE_SEND);
    cseg.qpn_ds = qpn_ds;
    cseg.signature_fm_ce_se = 0;
    cseg.fm_ce_se = i == count - 1 ? DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE
                                   : DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_FIRST_CQE_ERROR;
    cseg.imm = 0;
    // Ethernet segment without inline headers: cs_flags is byte 4
    uint64_t eseg[2] = {static_cast<uint64_t>(TX_CS_FLAGS) << 32, 0};
    struct mlx5_wqe_data_seg dseg;
    dseg.byte_count = doca_gpu_dev_verbs_bswap32(static_cast<uint32_t>(len));
    dseg.lkey = args.lkey;
    dseg.addr = doca_gpu_dev_verbs_bswap64(addr);

    struct doca_gpu_dev_verbs_wqe* wqe =
        doca_gpu_dev_verbs_get_wqe_ptr(qp, static_cast<uint16_t>(wqe_idx));
    doca_gpu_dev_verbs_store_wqe_seg(reinterpret_cast<uint64_t*>(&wqe->dseg0),
                                     reinterpret_cast<uint64_t*>(&cseg));
    doca_gpu_dev_verbs_store_wqe_seg(reinterpret_cast<uint64_t*>(&wqe->dseg1), eseg);
    doca_gpu_dev_verbs_store_wqe_seg(reinterpret_cast<uint64_t*>(&wqe->dseg2),
                                     reinterpret_cast<uint64_t*>(&dseg));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // The doorbell orders the work requests of every thread before it
    doca_gpu_dev_verbs_submit<SHARING, SCOPE, HANDLER, DOCA_GPUNETIO_VERBS_QP_SQ>(
        qp, wqe_first + count);
  }
}

// All threads: posts a burst, waiting for send queue room when needed. Returns false on a
// completion error.
__device__ bool tx_post_burst(const TxKernelArgs& args, const CqView& cq, TxRun& run,
                              uint64_t slot_start, uint32_t num_pkts) {
  __shared__ uint32_t ok;
  const uint64_t sq_size = doca_gpu_dev_verbs_load_const(&args.qp->sq_wqe_num);

  for (uint32_t done = 0; done < num_pkts;) {
    const uint32_t count = min(num_pkts - done, args.max_chunk);
    if (threadIdx.x == 0) {
      bool progress = false;
      bool room_ok = true;
      while (room_ok && (run.next_wqe + count - run.done_wqe > sq_size ||
                         run.chunks - run.cqes >= TX_CHUNKS_IN_FLIGHT)) {
        room_ok = tx_complete_one(args, cq, run, true, &progress);
      }
      ok = room_ok ? 1 : 0;
    }
    __syncthreads();
    if (ok == 0) {
      return false;
    }

    tx_post_chunk(args, run.next_wqe, slot_start + done, count);
    if (threadIdx.x == 0) {
      const uint32_t slot = run.chunks % TX_CHUNKS_IN_FLIGHT;
      run.chunk_end[slot] = run.next_wqe + count;
      run.chunk_last[slot] = (done + count == num_pkts) ? 1 : 0;
      run.next_wqe += count;
      run.chunks++;
    }
    __syncthreads();
    done += count;
  }
  return true;
}

// One block per TX queue of the GPU: block i serves queues[i] for the lifetime of the engine,
// several bursts in flight
__global__ void __launch_bounds__(TX_THREADS) tx_kernel(const TxKernelArgs* queues) {
  const TxKernelArgs args = queues[blockIdx.x];
  const CqView cq = cq_view(doca_gpu_dev_verbs_qp_get_cq_sq(args.qp));
  __shared__ TxRun run;
  __shared__ uint64_t slot_start;
  __shared__ uint32_t num_pkts;
  __shared__ uint32_t stop;

  uint64_t next_desc = 0;  // Thread 0 only

  if (threadIdx.x == 0) {
    run.next_wqe = 0;
    run.done_wqe = 0;
    run.chunks = 0;
    run.cqes = 0;
    run.bursts_done = 0;
    stop = 0;
  }
  __syncthreads();

  while (true) {
    // Thread 0 updates the shared state only once every thread is done with it
    __syncthreads();
    if (threadIdx.x == 0) {
      bool progress = true;
      bool ok = true;
      while (ok && progress) {
        ok = tx_complete_one(args, cq, run, false, &progress);
      }
      num_pkts = 0;
      if (!ok || load_acquire_sys(&args.ctrl->exit) != 0) {
        stop = 1;
      } else {
        TxBurstDesc* desc = &args.desc[next_desc & args.desc_mask];
        if (load_acquire_sys(&desc->ready) == static_cast<uint32_t>(next_desc + 1)) {
          slot_start = desc->slot_start;
          num_pkts = desc->num_pkts;
          next_desc++;
        }
      }
    }
    __syncthreads();
    if (stop != 0) {
      break;
    }
    if (num_pkts > 0 && !tx_post_burst(args, cq, run, slot_start, num_pkts)) {
      break;
    }
  }
}

// Blocks of the other kernels that share the GPU take part of every SM: a kernel that fits one SM
// k times uses 1/k of it per block, so the blocks of all the kernels fit if these fractions add up
// to at most the SM count
template <typename Kernel>
cudaError_t add_sm_share(Kernel kernel, uint32_t threads, uint32_t blocks, double* sms_needed) {
  if (blocks == 0) {
    return cudaSuccess;
  }
  int per_sm = 0;
  const cudaError_t err =
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, 0);
  if (err != cudaSuccess) {
    return err;
  }
  if (per_sm <= 0) {
    return cudaErrorLaunchOutOfResources;
  }
  *sms_needed += static_cast<double>(blocks) / per_sm;
  return cudaSuccess;
}

}  // namespace

uint32_t rx_kernel_threads() {
  return RX_THREADS;
}

cudaError_t load_kernels() {
  cudaFuncAttributes attr;
  cudaError_t err = cudaFuncGetAttributes(&attr, rx_kernel);
  if (err == cudaSuccess) {
    err = cudaFuncGetAttributes(&attr, tx_kernel);
  }
  return err;
}

cudaError_t check_gpu_capacity(uint32_t rx_blocks, uint32_t tx_blocks) {
  int device = 0;
  int sms = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  }
  double needed = 0;
  if (err == cudaSuccess) {
    err = add_sm_share(rx_kernel, RX_THREADS, rx_blocks, &needed);
  }
  if (err == cudaSuccess) {
    err = add_sm_share(tx_kernel, TX_THREADS, tx_blocks, &needed);
  }
  if (err != cudaSuccess) {
    return err;
  }
  return needed <= static_cast<double>(sms) ? cudaSuccess : cudaErrorLaunchOutOfResources;
}

cudaError_t launch_rx_kernel(const RxKernelArgs* queues, uint32_t num_queues, cudaStream_t stream) {
  rx_kernel<<<num_queues, RX_THREADS, 0, stream>>>(queues);
  return cudaGetLastError();
}

cudaError_t launch_tx_kernel(const TxKernelArgs* queues, uint32_t num_queues, cudaStream_t stream) {
  tx_kernel<<<num_queues, TX_THREADS, 0, stream>>>(queues);
  return cudaGetLastError();
}

}  // namespace daqiri::gpunetio_verbs
