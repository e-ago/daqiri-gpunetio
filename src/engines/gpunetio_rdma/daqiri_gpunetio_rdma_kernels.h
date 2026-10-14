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

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

struct doca_gpu_dev_verbs_qp;

// Interface between the gpunetio_rdma engine and its CUDA kernel. Each connection slot owns a DOCA
// Verbs RC queue pair exported to the GPU; one block of a resident kernel drives it. The host
// publishes send and receive bursts on descriptor rings in pinned host memory, the block posts
// their work requests, polls the completion queues and reports finished bursts through counters in
// the same memory.
namespace daqiri::gpunetio_rdma {

// Bursts in flight per direction on a slot. A power of two.
constexpr uint32_t DESC_RING = 256;

// Slot state. The host moves IDLE -> ACTIVE -> CLOSING, the kernel moves CLOSING -> DRAINED once
// every posted work request has completed and the GPU side of the queue pair is reset, and the
// host moves DRAINED -> IDLE after resetting the queue pair.
enum SlotState : uint32_t {
  SLOT_IDLE = 0,
  SLOT_ACTIVE = 1,
  SLOT_CLOSING = 2,
  SLOT_DRAINED = 3,
};

// Completion status of a burst
enum BurstStatus : uint32_t {
  BURST_OK = 0,
  BURST_ERROR = 1,    // a work request of the burst completed with an error
  BURST_FLUSHED = 2,  // the slot closed before the burst was posted
};

// Burst flags
constexpr uint32_t DESC_FLAG_GPU_BUFFERS = 1u << 0;  // receive buffers are in GPU memory

// One message of a burst. For receives the kernel replaces len with the received byte count.
struct Packet {
  uint64_t addr;
  uint32_t len;
  uint32_t lkey;  // big endian, as the DOCA WQE functions take memory keys
};

// One burst published by the host. ready is the burst index + 1 once the entry is valid.
struct BurstDesc {
  uint64_t ready;
  uint32_t first_pkt;  // first entry in the packet ring
  uint32_t num_pkts;
  uint32_t flags;
  uint32_t status;  // BurstStatus, written by the kernel before it counts the burst done
};

struct SlotCtrl {
  uint32_t state;  // SlotState
  uint32_t exit;   // set by the host to stop the kernel
  uint32_t error;  // first error completion: (direction << 16) | (syndrome << 8) | vendor syndrome
  uint32_t reserved;
  uint64_t sq_done;  // send bursts completed, in publishing order
  uint64_t rq_done;  // receive bursts completed, in publishing order
};

// Error completion directions in SlotCtrl::error
constexpr uint32_t ERROR_SEND = 1;
constexpr uint32_t ERROR_RECV = 2;

// Arguments of one block, in pinned host memory. All the pointers stay valid while the kernel runs.
struct SlotArgs {
  struct doca_gpu_dev_verbs_qp* qp;
  SlotCtrl* ctrl;
  BurstDesc* sq_desc;  // DESC_RING entries
  BurstDesc* rq_desc;  // DESC_RING entries
  Packet* sq_pkts;     // sq_pkt_mask + 1 entries
  Packet* rq_pkts;     // rq_pkt_mask + 1 entries
  uint32_t sq_pkt_mask;
  uint32_t rq_pkt_mask;
  uint64_t dump_addr;  // registered GPU buffer for the memory consistency WQE of pre-Hopper GPUs
  uint32_t dump_lkey;  // big endian
  uint32_t reserved;
};

// Threads of a block; they write the work requests of a burst in parallel
uint32_t kernel_threads();

// Loads the kernel on the current CUDA device. With CUDA lazy loading, the default, a kernel
// launched for the first time while a resident kernel runs doesn't start until that kernel exits.
cudaError_t load_kernels();

// Checks that the current GPU can run num_slots blocks at once: a block that finds no room would
// wait until shutdown and leave its slot unserved.
cudaError_t check_capacity(uint32_t num_slots);

// Starts the resident kernel, block i serving slots[i]. slots is in memory the GPU reads and stays
// valid while the kernel runs.
cudaError_t launch_kernel(const SlotArgs* slots, uint32_t num_slots, cudaStream_t stream);

}  // namespace daqiri::gpunetio_rdma
