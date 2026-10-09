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

// Interface between the gpunetio engine and its CUDA kernels. The kernels drive the NIC queues
// through DOCA GPUNetIO; the engine exchanges bursts with them through rings in pinned host
// memory, which both the CPU and the GPU access.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

struct doca_gpu_eth_rxq;
struct doca_gpu_eth_txq;

namespace daqiri::gpunetio {

// Error a kernel reports before it stops
enum KernelError : uint32_t {
  KERNEL_OK = 0,
  KERNEL_RX_ERROR_CQE = 1,     // receive completion with error
  KERNEL_RX_OUT_OF_ORDER = 2,  // packets don't follow the receive ring order
  KERNEL_TX_ERROR_CQE = 3,     // send completion with error
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
  uint32_t pad1;
  uint64_t pkts;    // Packets received so far
  uint64_t bursts;  // Bursts published so far
};

struct RxKernelArgs {
  doca_gpu_eth_rxq* rxq;
  RxBurstDesc* desc;    // Pinned host memory, desc_mask + 1 entries
  uint32_t* pkt_len;    // Pinned host memory, one entry per ring slot
  RxControl* ctrl;      // Pinned host memory
  uint64_t ring_mask;   // Packets in the receive ring - 1 (power of two)
  uint64_t timeout_ns;  // Publish a partial burst after this time, 0 never
  uint32_t desc_mask;
  uint32_t batch_size;  // Packets per burst
  uint32_t max_pkts;    // Packets per receive call, a multiple of rx_kernel_threads()
};

// Packet of a TX burst, one entry per TX slot
struct TxPacket {
  uint64_t addr;
  uint32_t len;
  uint32_t pad;
};

// Burst to send, published by the CPU to the persistent TX kernel
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
  uint32_t pad1;
  uint64_t completed_bursts;  // Bursts the NIC has sent so far
};

// TX queue state kept in device memory between per-burst kernel launches
struct TxState {
  uint64_t next_wqe;     // Index of the next send WQE
  uint64_t done_wqe;     // WQEs the NIC has completed
  uint64_t chunks;       // Chunks posted, each one asks for a completion
  uint64_t bursts_done;  // Bursts completed
};

struct TxKernelArgs {
  doca_gpu_eth_txq* txq;
  const TxPacket* pkts;  // Pinned host memory, one entry per TX slot
  TxBurstDesc* desc;     // Pinned host memory, desc_mask + 1 entries (persistent kernel)
  TxControl* ctrl;       // Pinned host memory
  TxState* state;        // Device memory
  uint64_t num_slots;
  uint32_t desc_mask;
  uint32_t mkey;       // Memory key of the TX slots, network byte order
  uint32_t max_chunk;  // Packets per completion, at most half of the send queue
};

// Threads of the RX kernel block
uint32_t rx_kernel_threads();

cudaError_t launch_rx_kernel(const RxKernelArgs& args, cudaStream_t stream);
cudaError_t launch_tx_persistent_kernel(const TxKernelArgs& args, cudaStream_t stream);
cudaError_t launch_tx_burst_kernel(const TxKernelArgs& args, uint64_t slot_start, uint32_t num_pkts,
                                   cudaStream_t stream);

}  // namespace daqiri::gpunetio
