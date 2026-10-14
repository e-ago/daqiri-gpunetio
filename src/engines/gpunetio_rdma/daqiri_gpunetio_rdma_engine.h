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

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <daqiri/daqiri.h>

#include "src/daqiri_pool.h"
#include "src/daqiri_ring.h"
#include "src/engine.h"

// DOCA, libibverbs and librdmacm types stay opaque, so daqiri_common needs none of their headers
struct doca_gpu;
struct ibv_context;
struct rdma_cm_id;
struct rdma_event_channel;
struct rdma_cm_event;

namespace daqiri {

/**
 * Experimental RoCE engine on DOCA Verbs and DOCA GPUNetIO (stream_type: socket with roce://
 * endpoints, engine: gpunetio_verbs). As the user-facing ibverbs name covers the rdma and ibverbs
 * engines, gpunetio_verbs covers this engine for socket streams and the raw gpunetio_verbs engine
 * for raw streams. It keeps the connection model and the API of the ibverbs RDMA engine: rdma_cm
 * connects clients and servers, and the application posts SEND and RECEIVE bursts on a connection
 * and gets them back as completions. The NIC queues are driven from the GPU instead: each queue
 * slot owns a DOCA Verbs RC queue pair, created at initialization with its rings in GPU memory and
 * exported to the GPU, and one block of a resident CUDA kernel posts the work requests of the slot
 * and polls its completions. One CPU thread per slot moves bursts between the application and the
 * kernel. A slot serves one connection at a time and is reused after a disconnect.
 */
class GpunetioRdmaEngine : public Engine {
 public:
  GpunetioRdmaEngine();
  ~GpunetioRdmaEngine() override;

  bool set_config_and_initialize(const NetworkConfig& cfg) override;
  void initialize() override;
  void run() override;
  void shutdown() override;
  void print_stats() override;
  bool validate_config() const override;

  // Packet accessors
  void* get_packet_ptr(BurstParams* burst, int idx) override;
  uint32_t get_packet_length(BurstParams* burst, int idx) override;
  void* get_segment_packet_ptr(BurstParams* burst, int seg, int idx) override;
  uint32_t get_segment_packet_length(BurstParams* burst, int seg, int idx) override;
  FlowId get_packet_flow_id(BurstParams* burst, int idx) override;
  Status get_packet_rx_timestamp(BurstParams* burst, int idx, uint64_t* timestamp_ns) override;
  void* get_packet_extra_info(BurstParams* burst, int idx) override;
  uint64_t get_burst_tot_byte(BurstParams* burst) override;

  // Bursts
  BurstParams* create_tx_burst_params() override;
  Status get_tx_packet_burst(BurstParams* burst) override;
  bool is_tx_burst_available(BurstParams* burst) override;
  Status set_packet_lengths(BurstParams* burst, int idx,
                            const std::initializer_list<int>& lens) override;
  Status set_packet_tx_time(BurstParams* burst, int idx, uint64_t time) override;
  Status set_eth_header(BurstParams* burst, int idx, char* dst_addr) override;
  Status set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                         unsigned int src_host, unsigned int dst_host) override;
  Status set_udp_header(BurstParams* burst, int idx, int udp_len, uint16_t src_port,
                        uint16_t dst_port) override;
  Status set_udp_payload(BurstParams* burst, int idx, void* data, int len) override;
  Status send_tx_burst(BurstParams* burst) override;
  Status get_rx_burst(BurstParams** burst, int port, int q) override;
  Status get_rx_burst(BurstParams** burst, uintptr_t conn_id, bool server) override;
  void free_all_segment_packets(BurstParams* burst, int seg) override;
  void free_all_packets(BurstParams* burst) override;
  void free_packet_segment(BurstParams* burst, int seg, int pkt) override;
  void free_packet(BurstParams* burst, int pkt) override;
  void free_rx_burst(BurstParams* burst) override;
  void free_tx_burst(BurstParams* burst) override;
  void free_rx_metadata(BurstParams* burst) override;
  void free_tx_metadata(BurstParams* burst) override;
  Status get_mac_addr(int port, char* mac) override;

  // Connections
  Status rdma_connect_to_server(const std::string& dst_addr, uint16_t dst_port,
                                uintptr_t* conn_id) override;
  Status rdma_connect_to_server(const std::string& dst_addr, uint16_t dst_port,
                                const std::string& src_addr, uintptr_t* conn_id) override;
  Status rdma_get_port_queue(uintptr_t conn_id, uint16_t* port, uint16_t* queue) override;
  Status rdma_get_server_conn_id(const std::string& server_addr, uint16_t server_port,
                                 uintptr_t* conn_id) override;
  Status rdma_set_header(BurstParams* burst, RDMAOpCode op_code, uintptr_t conn_id, bool is_server,
                         int num_pkts, uint64_t wr_id, const std::string& local_mr_name) override;
  RDMAOpCode rdma_get_opcode(BurstParams* burst) override;

  struct Device;
  struct Slot;
  struct Listener;

 private:
  bool setup_devices();
  bool register_memory_regions();
  bool setup_gpu();
  bool create_slots();
  bool create_queue_pair(Slot& slot);
  void destroy_queue_pair(Slot& slot);
  bool setup_pools();
  bool start_kernel();
  bool stop_kernel();
  bool start_listeners();
  void destroy_resources();

  bool transition_qp(Slot& slot, struct rdma_cm_id* id, int state);
  bool reset_qp(Slot& slot);
  bool set_qp_error(Slot& slot);
  void handle_connect_request(struct rdma_cm_event* event);
  void handle_established(struct rdma_cm_id* id);
  bool handle_disconnect(struct rdma_cm_id* id);
  void release_connecting_slot(Slot& slot);
  void activate_slot(Slot& slot);

  void worker(Slot* slot);
  bool publish(Slot& slot, BurstParams* burst);
  void reap(Slot& slot);
  void complete(Slot& slot, BurstParams* burst, Status status);
  void flush_completions(Slot& slot);
  void close_slot(Slot& slot);
  void free_burst_internal(BurstParams* burst);

  Slot* find_slot(uintptr_t conn_id);
  Slot* find_slot_by_cm_id(struct rdma_cm_id* id);
  Slot* claim_free_slot(int if_idx, bool server);
  uint32_t lkey_for(const Device& dev, const std::string& mr_name) const;

  std::vector<std::unique_ptr<Device>> devices_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::vector<std::unique_ptr<Listener>> listeners_;
  std::vector<Device*> if_device_;  // interface index -> device

  // GPU that runs the kernel and holds the queue rings
  int gpu_ordinal_ = 0;
  struct doca_gpu* gpu_ = nullptr;
  cudaStream_t kernel_stream_ = nullptr;
  bool kernel_running_ = false;
  bool kernel_stuck_ = false;
  void* slot_args_ = nullptr;  // gpunetio_rdma::SlotArgs[], pinned

  // Connection manager. The device list keeps the RDMA device contexts open.
  struct ibv_context** cm_devices_ = nullptr;
  struct rdma_event_channel* cm_channel_ = nullptr;
  std::thread cm_thread_;
  std::mutex slots_mutex_;  // slot claiming and connection state changes

  std::atomic<bool> workers_running_{false};
  std::atomic<bool> cm_running_{false};

  // Burst metadata and buffers. Completions return the burst objects the application posted, so
  // TX and RX metadata share one pool.
  daqiri::ObjectPool* meta_pool_ = nullptr;
  daqiri::ObjectPool* pkt_ptr_pool_ = nullptr;
  daqiri::ObjectPool* pkt_len_pool_ = nullptr;
  std::unordered_map<std::string, daqiri::Ring*> buf_pools_;
};

}  // namespace daqiri
