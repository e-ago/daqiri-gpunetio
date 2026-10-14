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

// Interface between the gpunetio_verbs engine and its CUDA kernels. The kernels drive DOCA Verbs
// Ethernet queues through the DOCA GPUNetIO device functions; the engine exchanges bursts with
// them through rings in pinned host memory, which both the CPU and the GPU access.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

struct doca_gpu_dev_verbs_qp;

namespace daqiri::gpunetio_verbs {

// Error a kernel reports before it stops
enum KernelError : uint32_t {
  KERNEL_OK = 0,
  KERNEL_RX_ERROR_CQE = 1,     // receive completion with error, syndrome in the control block
  KERNEL_RX_OUT_OF_ORDER = 2,  // completion of an unexpected receive work request
  KERNEL_TX_ERROR_CQE = 3,     // send completion with error, syndrome in the control block
  KERNEL_TX_OUT_OF_ORDER = 4,  // completion of an unexpected send work request
};

// Burst of received packets, published by the RX kernel. The packets are consecutive in the
// receive ring and never wrap around its end.
struct alignas(16) RxBurstDesc {
  uint64_t pkt_seq;  // Count of packets received before this burst; ring slot = pkt_seq % ring size
  uint32_t num_pkts;
  uint32_t ready;  // Descriptor index + 1, written last
};

struct alignas(64) RxControl {
  // Written by the CPU
  uint32_t exit;
  uint32_t pad0;
  uint64_t released_pkts;  // Packets released by the application so far, oldest first
  // Written by the GPU
  uint32_t error;
  uint32_t syndrome;
  uint64_t pkts;    // Packets received so far
  uint64_t bursts;  // Bursts published so far
};

struct RxKernelArgs {
  doca_gpu_dev_verbs_qp* qp;  // Receive queue and its completion queue, exported to the GPU
  RxBurstDesc* desc;          // Pinned host memory, desc_mask + 1 entries
  uint32_t* pkt_len;          // Pinned host memory, one entry per ring slot
  uint32_t* flow_tag;         // Pinned host memory, one entry per ring slot
  RxControl* ctrl;            // Pinned host memory
  uint64_t timeout_ns;        // Publish a partial burst after this time, 0 never
  uint32_t ring_mask;         // Packets in the receive ring - 1 (power of two)
  uint32_t desc_mask;
  uint32_t batch_size;  // Packets per burst
  uint32_t max_pkts;    // Completions per polling round, a multiple of rx_kernel_threads()
};

// Packet of a TX burst, one entry per TX slot
struct TxPacket {
  uint64_t addr;
  uint64_t len;
};

// Burst to send, published by the CPU to the TX kernel
struct alignas(16) TxBurstDesc {
  uint64_t slot_start;  // First TX slot, slot index = slot_start % slot count
  uint32_t num_pkts;
  uint32_t ready;  // Descriptor index + 1, written last
};

struct alignas(64) TxControl {
  // Written by the CPU
  uint32_t exit;
  uint32_t pad0;
  // Written by the GPU
  uint32_t error;
  uint32_t syndrome;
  uint64_t completed_bursts;  // Bursts the NIC has sent so far
};

struct TxKernelArgs {
  doca_gpu_dev_verbs_qp* qp;  // Send queue and its completion queue, exported to the GPU
  const TxPacket* pkts;       // Pinned host memory, one entry per TX slot
  TxBurstDesc* desc;          // Pinned host memory, desc_mask + 1 entries
  TxControl* ctrl;            // Pinned host memory
  uint64_t num_slots;
  uint32_t desc_mask;
  uint32_t lkey;       // Memory key of the TX slots, network byte order
  uint32_t max_chunk;  // Packets per completion, at most half of the send queue
  uint32_t pad;
};

// Threads of an RX kernel block
uint32_t rx_kernel_threads();

// Loads every kernel on the current CUDA device. With CUDA lazy loading, the default, a kernel
// launched for the first time while a resident kernel runs doesn't start until that kernel exits,
// so call it before the first resident kernel starts.
cudaError_t load_kernels();

// Checks that rx_blocks RX blocks and tx_blocks TX blocks can run on the current GPU at once.
// Returns cudaErrorLaunchOutOfResources if not, as a resident block that finds no room would wait
// until shutdown and leave its queue unserved.
cudaError_t check_gpu_capacity(uint32_t rx_blocks, uint32_t tx_blocks);

// The resident kernels run one block per queue, block i serving queues[i]. queues holds
// num_queues entries in memory the GPU reads, and stays valid while the kernel runs.
cudaError_t launch_rx_kernel(const RxKernelArgs* queues, uint32_t num_queues, cudaStream_t stream);
cudaError_t launch_tx_kernel(const TxKernelArgs* queues, uint32_t num_queues, cudaStream_t stream);

}  // namespace daqiri::gpunetio_verbs
