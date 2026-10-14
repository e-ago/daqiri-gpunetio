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

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "src/daqiri_pool.h"
#include "src/engine.h"
#include <daqiri/daqiri.h>

struct doca_gpu;

namespace daqiri {

/**
 * @brief Raw-Ethernet engine built on DOCA Verbs and DOCA GPUNetIO (stream_type: raw, engine:
 * gpunetio_verbs)
 *
 * The queues are DOCA Verbs Ethernet send and receive queues whose rings, doorbell records and
 * completion queues live in GPU memory, exported to the GPU with DOCA GPUNetIO. CUDA kernels drive
 * them: on the GPU of the engine, one resident RX kernel serves all the RX queues and one resident
 * TX kernel all the TX queues, one block per queue. An RX block receives into a ring of slots in
 * its queue's memory region and publishes bursts; the slots of a burst go back to the NIC only when
 * the application frees it. A TX block sends bursts from cyclic slots of its queue's memory region.
 * One CPU thread per cpu_core moves the bursts of its queues between the kernels and the
 * application. RX flows are steered with mlx5 DevX TIRs and mlx5dv_dr rules on the receive queues.
 */
class GpunetioVerbsEngine : public Engine {
 public:
  GpunetioVerbsEngine();
  ~GpunetioVerbsEngine() override;

  bool set_config_and_initialize(const NetworkConfig& cfg) override;
  void initialize() override;
  void run() override;
  void shutdown() override;
  void print_stats() override;
  Status get_mac_addr(int port, char* mac) override;

  // Packet accessors
  void* get_packet_ptr(BurstParams* burst, int idx) override;
  uint32_t get_packet_length(BurstParams* burst, int idx) override;
  void* get_segment_packet_ptr(BurstParams* burst, int seg, int idx) override;
  uint32_t get_segment_packet_length(BurstParams* burst, int seg, int idx) override;
  FlowId get_packet_flow_id(BurstParams* burst, int idx) override;
  Status get_packet_rx_timestamp(BurstParams* burst, int idx, uint64_t* timestamp_ns) override;
  void* get_packet_extra_info(BurstParams* burst, int idx) override;

  // TX
  BurstParams* create_tx_burst_params() override;
  bool is_tx_burst_available(BurstParams* burst) override;
  Status get_tx_packet_burst(BurstParams* burst) override;
  Status set_eth_header(BurstParams* burst, int idx, char* dst_addr) override;
  Status set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                         unsigned int src_host, unsigned int dst_host) override;
  Status set_udp_header(BurstParams* burst, int idx, int udp_len, uint16_t src_port,
                        uint16_t dst_port) override;
  Status set_udp_payload(BurstParams* burst, int idx, void* data, int len) override;
  Status set_packet_lengths(BurstParams* burst, int idx,
                            const std::initializer_list<int>& lens) override;
  Status set_packet_tx_time(BurstParams* burst, int idx, uint64_t time) override;
  Status send_tx_burst(BurstParams* burst) override;

  // RX
  Status get_rx_burst(BurstParams** burst, int port, int q) override;

  // Free family
  void free_all_segment_packets(BurstParams* burst, int seg) override;
  void free_all_packets(BurstParams* burst) override;
  void free_packet_segment(BurstParams* burst, int seg, int pkt) override;
  void free_packet(BurstParams* burst, int pkt) override;
  void free_rx_burst(BurstParams* burst) override;
  void free_tx_burst(BurstParams* burst) override;
  void free_rx_metadata(BurstParams* burst) override;
  void free_tx_metadata(BurstParams* burst) override;
  uint64_t get_burst_tot_byte(BurstParams* burst) override;

 private:
  struct Device;
  struct Destination;
  struct RxQueue;
  struct TxQueue;

  bool setup_gpu();
  bool open_devices();
  bool size_memory_regions();
  bool register_memory_regions();
  bool create_rx_queue(RxQueue& q);
  bool create_tx_queue(TxQueue& q);
  bool program_steering(Device& dev, const InterfaceConfig& intf);
  Destination* rx_destination(Device& dev, const std::vector<RxQueue*>& queues);
  bool create_burst_pools();
  bool start_kernels();
  void stop_kernels();
  void destroy_steering(Device& dev);

  void worker(int core, std::vector<RxQueue*> rx, std::vector<TxQueue*> tx);
  bool serve_rx_queue(RxQueue* q);
  bool serve_tx_queue(TxQueue* q);

  RxQueue* find_rx_queue(int port, int q) const;
  TxQueue* find_tx_queue(int port, int q) const;
  Status write_packet(BurstParams* burst, int idx, size_t offset, const void* data, size_t len);

  int gpu_ordinal_ = 0;
  struct doca_gpu* gpu_ = nullptr;
  // Before Hopper, GPUDirect RDMA writes may reach other GPU clients out of order: flush them
  // before the application reads received packets in GPU memory
  bool flush_rx_writes_ = false;

  std::vector<std::unique_ptr<Device>> devices_;
  std::vector<Device*> port_devices_;  // Indexed by port id
  std::vector<std::unique_ptr<RxQueue>> rx_queues_;
  std::vector<std::unique_ptr<TxQueue>> tx_queues_;

  // Resident kernels, each on its own stream: on a shared stream, the second would wait for the
  // first to exit
  void* rx_args_ = nullptr;  // Pinned host memory, one entry per block
  void* tx_args_ = nullptr;
  cudaStream_t rx_stream_ = nullptr;
  cudaStream_t tx_stream_ = nullptr;
  bool kernels_stuck_ = false;

  daqiri::ObjectPool* rx_meta_pool_ = nullptr;
  daqiri::ObjectPool* tx_meta_pool_ = nullptr;
  std::vector<BurstParams*> burst_objects_;  // Constructed in the pools, destroyed at shutdown
  uint32_t max_tx_batch_ = 0;

  std::atomic<bool> workers_running_{false};
  std::vector<std::thread> workers_;  // One per cpu_core of the queues
};

}  // namespace daqiri
