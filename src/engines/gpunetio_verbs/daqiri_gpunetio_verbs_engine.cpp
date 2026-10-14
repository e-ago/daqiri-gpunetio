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

#include "src/engines/gpunetio_verbs/daqiri_gpunetio_verbs_engine.h"

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_log.h>
#include <doca_rdma_bridge.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <endian.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include "src/daqiri_ring.h"
#include "src/engines/gpunetio_verbs/daqiri_gpunetio_verbs_kernels.h"
#include "src/engines/gpunetio_verbs/daqiri_gpunetio_verbs_prm.h"
#include "src/rss.h"
#include <daqiri/logging.hpp>

namespace daqiri {

namespace gv = gpunetio_verbs;

namespace {

constexpr uint64_t RX_RING_MIN_PKTS = 64;
// The GPU queue handles of DOCA GPUNetIO count work requests in 16 bits
constexpr uint64_t RX_RING_MAX_PKTS = 32768;
// Completions per polling round of the RX kernel
constexpr uint32_t RX_MAX_PKTS_PER_ROUND = 4096;
// Bursts in flight on a TX queue
constexpr uint32_t TX_DESC_RING = 1024;
constexpr uint32_t TX_SEND_RING = 4096;
constexpr uint64_t TX_SQ_MIN_WQES = 1024;
constexpr uint64_t TX_SQ_MAX_WQES = 32768;
// The TX kernel asks for a completion at least every 1/8 of the send queue
constexpr uint32_t TX_CHUNKS_PER_SQ = 8;
constexpr size_t RQ_WQE_SIZE = 16;
constexpr size_t SQ_WQE_SIZE = 64;
constexpr size_t CQE_SIZE = 64;
constexpr size_t CQ_DBR_SIZE = 8;
// The UAR doorbell register the GPU writes
constexpr size_t UAR_DB_SIZE = 8;
constexpr size_t SLOT_ALIGN = 64;
// A steering rule tags its packets with the flow ID in 24 bits of the completion
constexpr FlowId MAX_FLOW_TAG = 0x00ffffffU;
constexpr auto KERNEL_STOP_TIMEOUT = std::chrono::seconds(5);

constexpr uint32_t GV_RX_BURST_FLAG = 1u << 25;
constexpr uint32_t GV_TX_BURST_FLAG = 1u << 26;
constexpr uint32_t BURST_MAGIC = 0x67767262u;

// Engine data of a burst, kept in its custom header area
struct BurstPriv {
  uint64_t seq;  // RX: descriptor index; TX: first TX slot
  void* queue;   // RxQueue or TxQueue of the burst
  uint32_t magic;
  uint32_t reserved;  // TX: number of slots reserved from seq and not retired yet
};
static_assert(sizeof(BurstPriv) <= sizeof(BurstHeader::custom_burst_data),
              "BurstPriv must fit in BurstHeader::custom_burst_data");

void set_priv(BurstParams* burst, uint64_t seq, void* queue, uint32_t reserved = 0) {
  const BurstPriv priv{seq, queue, BURST_MAGIC, reserved};
  std::memcpy(burst->hdr.custom_burst_data, &priv, sizeof(priv));
}

bool get_priv(const BurstParams* burst, BurstPriv* priv) {
  std::memcpy(priv, burst->hdr.custom_burst_data, sizeof(*priv));
  return priv->magic == BURST_MAGIC && priv->queue != nullptr;
}

uint64_t next_pow2(uint64_t value) {
  uint64_t pow = 1;
  while (pow < value) {
    pow <<= 1;
  }
  return pow;
}

uint64_t prev_pow2(uint64_t value) {
  uint64_t pow = 1;
  while (pow * 2 <= value) {
    pow <<= 1;
  }
  return pow;
}

uint64_t round_up(uint64_t value, uint64_t align) {
  return (value + align - 1) / align * align;
}

size_t page_size() {
  static const size_t size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return size;
}

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

int parse_core(const std::string& core) {
  if (core.empty()) {
    return -1;
  }
  return static_cast<int>(std::strtol(core.c_str(), nullptr, 10));
}

void pin_thread(int core, const std::string& name) {
  if (core < 0) {
    return;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    DAQIRI_LOG_WARN("Could not pin the {} thread to core {}", name, core);
  }
}

// Counters and flags in pinned host memory, shared with the kernels
template <typename T>
T load_acquire(const T* ptr) {
  return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

template <typename T>
void store_release(T* ptr, T value) {
  __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

// Pinned host memory, mapped in the address space of every GPU
template <typename T>
T* alloc_pinned(size_t count) {
  void* ptr = nullptr;
  if (cudaHostAlloc(&ptr, count * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable) !=
      cudaSuccess) {
    return nullptr;
  }
  std::memset(ptr, 0, count * sizeof(T));
  return static_cast<T*>(ptr);
}

void free_pinned(void* ptr) {
  if (ptr != nullptr) {
    cudaFreeHost(ptr);
  }
}

const char* doca_err(doca_error_t err) {
  return doca_error_get_descr(err);
}

const char* kernel_error_str(uint32_t error) {
  switch (error) {
    case gv::KERNEL_RX_ERROR_CQE:
      return "receive completion with error (packet bigger than the ring slot?)";
    case gv::KERNEL_RX_OUT_OF_ORDER:
      return "completion of an unexpected receive work request";
    case gv::KERNEL_TX_ERROR_CQE:
      return "send completion with error";
    case gv::KERNEL_TX_OUT_OF_ORDER:
      return "completion of an unexpected send work request";
    default:
      return "unknown error";
  }
}

bool wait_stream(cudaStream_t stream) {
  const auto t0 = std::chrono::steady_clock::now();
  while (cudaStreamQuery(stream) == cudaErrorNotReady) {
    if (std::chrono::steady_clock::now() - t0 > KERNEL_STOP_TIMEOUT) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

uint32_t doca_sdk_log_level(LogLevel::Level level) {
  switch (level) {
    case LogLevel::TRACE:
      return DOCA_LOG_LEVEL_TRACE;
    case LogLevel::DEBUG:
      return DOCA_LOG_LEVEL_DEBUG;
    case LogLevel::ERROR:
      return DOCA_LOG_LEVEL_ERROR;
    case LogLevel::CRITICAL:
      return DOCA_LOG_LEVEL_CRIT;
    case LogLevel::OFF:
      return DOCA_LOG_LEVEL_DISABLE;
    default:
      // DOCA prints its own setup details at info level
      return DOCA_LOG_LEVEL_WARNING;
  }
}

// The DOCA SDK explains most failures only in its own log, which stays silent until a backend
// is registered. Backends live until the process exits, so the stderr backend is created once
// and its level follows the configuration of the last initialization.
void set_doca_sdk_log_level(LogLevel::Level level) {
  static std::mutex mutex;
  static struct doca_log_backend* backend = nullptr;
  std::lock_guard<std::mutex> lock(mutex);

  if (backend == nullptr) {
    const doca_error_t ret = doca_log_backend_create_with_file_sdk(stderr, &backend);
    if (ret != DOCA_SUCCESS) {
      backend = nullptr;
      DAQIRI_LOG_WARN("Could not route DOCA SDK logs to stderr: {}", doca_err(ret));
      return;
    }
  }
  const uint32_t doca_level = doca_sdk_log_level(level);
  (void)doca_log_level_set_global_sdk_limit(doca_level);
  (void)doca_log_backend_set_sdk_level(backend, doca_level);
}

bool is_pci_addr(const std::string& key) {
  return key.find(':') != std::string::npos && key.find('.') != std::string::npos;
}

// Matches a PCIe address ("0000:3b:00.0" or "3b:00.0"), an IB device name ("mlx5_0") or a
// netdev name
bool devinfo_matches(struct doca_devinfo* devinfo, const std::string& key) {
  if (key.empty()) {
    return false;
  }
  if (is_pci_addr(key)) {
    uint8_t is_equal = 0;
    return doca_devinfo_is_equal_pci_addr(devinfo, key.c_str(), &is_equal) == DOCA_SUCCESS &&
           is_equal != 0;
  }
  char name[DOCA_DEVINFO_IFACE_NAME_SIZE] = {};
  if (doca_devinfo_get_ibdev_name(devinfo, name, DOCA_DEVINFO_IBDEV_NAME_SIZE) == DOCA_SUCCESS &&
      key == name) {
    return true;
  }
  return doca_devinfo_get_iface_name(devinfo, name, sizeof(name)) == DOCA_SUCCESS && key == name;
}

// RX queues a flow steers to: the queue action ends the flow
std::vector<uint16_t> flow_queues(const FlowConfig& flow) {
  const FlowAction& action = flow.actions_.empty() ? flow.action_ : flow.actions_.back();
  if (!action.ids_.empty()) {
    return action.ids_;
  }
  return {action.id_};
}

std::pair<uint32_t, uint16_t> split_mac(const std::array<uint8_t, 6>& mac) {
  const uint32_t high = (static_cast<uint32_t>(mac[0]) << 24) |
                        (static_cast<uint32_t>(mac[1]) << 16) |
                        (static_cast<uint32_t>(mac[2]) << 8) | static_cast<uint32_t>(mac[3]);
  const uint16_t low = static_cast<uint16_t>((mac[4] << 8) | mac[5]);
  return {high, low};
}

// mlx5dv_dr match parameter holding the outer-header match set
struct DrMatch {
  size_t match_sz = sizeof(buf);
  uint64_t buf[8] = {};  // sizeof(fte_match_set_lyr_2_4)
};
static_assert(sizeof(DrMatch::buf) == DEVX_ST_SZ_BYTES(fte_match_set_lyr_2_4),
              "The match buffer holds the outer-header match set");

struct mlx5dv_flow_match_parameters* dr_param(DrMatch& match) {
  return reinterpret_cast<struct mlx5dv_flow_match_parameters*>(&match);
}

// A DOCA Verbs Ethernet queue's rings in GPU memory, and its handle on the GPU
struct QueueRings {
  void* wq = nullptr;  // Work queue
  struct doca_umem* wq_umem = nullptr;
  void* dbr = nullptr;  // Doorbell record
  struct doca_umem* dbr_umem = nullptr;
  void* cq_ring = nullptr;  // Completion queue and its doorbell record
  struct doca_umem* cq_umem = nullptr;
  struct doca_verbs_cq* cq = nullptr;
  uint32_t cq_size = 0;
  struct doca_uar* uar = nullptr;
  struct doca_gpu_verbs_qp* gpu_qp = nullptr;
  struct doca_gpu_dev_verbs_qp* dev_qp = nullptr;
};

// GPU memory for a queue ring, registered with the NIC
bool alloc_gpu_ring(struct doca_gpu* gpu, struct doca_dev* dev, size_t size, void** ptr,
                    struct doca_umem** umem, const std::string& what) {
  doca_error_t ret =
      doca_gpu_mem_alloc(gpu, size, page_size(), DOCA_GPU_MEM_TYPE_GPU, ptr, nullptr);
  if (ret != DOCA_SUCCESS) {
    *ptr = nullptr;
    DAQIRI_LOG_CRITICAL("Could not allocate {} bytes of GPU memory for the {}: {}", size, what,
                        doca_err(ret));
    return false;
  }
  if (cudaMemset(*ptr, 0, size) != cudaSuccess) {
    DAQIRI_LOG_CRITICAL("Could not clear the {}", what);
    return false;
  }
  ret = doca_umem_gpu_create(gpu, dev, *ptr, size,
                             DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE |
                                 DOCA_ACCESS_FLAG_RDMA_READ | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
                             umem);
  if (ret != DOCA_SUCCESS) {
    *umem = nullptr;
    DAQIRI_LOG_CRITICAL("Could not register the {} with the NIC: {}", what, doca_err(ret));
    return false;
  }
  return true;
}

// Work queue and doorbell record of a queue
bool alloc_work_queue(struct doca_gpu* gpu, struct doca_dev* dev, size_t wq_bytes,
                      QueueRings& rings, const std::string& name) {
  return alloc_gpu_ring(gpu, dev, round_up(wq_bytes, page_size()), &rings.wq, &rings.wq_umem,
                        "work queue of " + name) &&
         alloc_gpu_ring(gpu, dev, page_size(), &rings.dbr, &rings.dbr_umem,
                        "doorbell record of " + name);
}

// Completion queue with its CQEs and doorbell record in GPU memory. Overrun mode: the kernels track
// their position themselves and never have more completions pending than the queue holds.
bool create_cq(struct doca_gpu* gpu, struct doca_dev* dev, struct doca_verbs_context* vctx,
               uint32_t entries, QueueRings& rings, const std::string& name) {
  const size_t size = round_up(static_cast<size_t>(entries) * CQE_SIZE + CQ_DBR_SIZE, page_size());
  if (!alloc_gpu_ring(gpu, dev, size, &rings.cq_ring, &rings.cq_umem,
                      "completion queue of " + name)) {
    return false;
  }
  // Every CQE starts invalid and owned by the hardware
  std::vector<struct mlx5_cqe64> cqes(entries);
  std::memset(cqes.data(), 0, cqes.size() * sizeof(cqes[0]));
  for (auto& cqe : cqes) {
    cqe.op_own = static_cast<uint8_t>((MLX5_CQE_INVALID << 4) | MLX5_CQE_OWNER_MASK);
  }
  if (cudaMemcpy(rings.cq_ring, cqes.data(), cqes.size() * sizeof(cqes[0]),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    DAQIRI_LOG_CRITICAL("Could not initialize the completion queue of {}", name);
    return false;
  }
  struct doca_verbs_cq_attr* attr = nullptr;
  doca_error_t ret = doca_verbs_cq_attr_create(&attr);
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_attr_set_external_datapath_en(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_attr_set_external_umem(attr, rings.cq_umem, 0);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_attr_set_cq_size(attr, entries);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_attr_set_cq_overrun(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_create(vctx, attr, &rings.cq);
  }
  if (attr != nullptr) {
    doca_verbs_cq_attr_destroy(attr);
  }
  if (ret != DOCA_SUCCESS) {
    rings.cq = nullptr;
    DAQIRI_LOG_CRITICAL("Could not create the completion queue of {}: {}", name, doca_err(ret));
    return false;
  }
  rings.cq_size = entries;
  return true;
}

// The kernels ring the doorbells: a non-cached UAR, as the DOCA GPUNetIO samples use
bool create_uar(struct doca_dev* dev, QueueRings& rings, const std::string& name) {
  doca_error_t ret = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &rings.uar);
  if (ret != DOCA_SUCCESS) {
    ret = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, &rings.uar);
  }
  if (ret != DOCA_SUCCESS) {
    ret = doca_uar_create(dev, DOCA_UAR_ALLOCATION_TYPE_BLUEFLAME, &rings.uar);
  }
  if (ret != DOCA_SUCCESS) {
    rings.uar = nullptr;
    DAQIRI_LOG_CRITICAL("Could not create a UAR for {}: {}", name, doca_err(ret));
    return false;
  }
  return true;
}

// Where DOCA Verbs placed a completion queue
struct CqLayout {
  void* cqes = nullptr;
  uint32_t entries = 0;
  uint32_t* dbr = nullptr;
};

CqLayout cq_layout(struct doca_verbs_cq* cq) {
  CqLayout layout;
  uint8_t entry_size = 0;
  doca_verbs_cq_get_wq(cq, &layout.cqes, &layout.entries, &entry_size);
  uint64_t* uar_db = nullptr;
  uint32_t* arm_dbr = nullptr;
  doca_verbs_cq_get_dbr_addr(cq, &uar_db, &layout.dbr, &arm_dbr);
  return layout;
}

void destroy_rings(struct doca_gpu* gpu, QueueRings& rings) {
  if (rings.gpu_qp != nullptr) {
    doca_gpu_verbs_unexport_qp(gpu, rings.gpu_qp);
    rings.gpu_qp = nullptr;
    rings.dev_qp = nullptr;
  }
  if (rings.cq != nullptr) {
    doca_verbs_cq_destroy(rings.cq);
    rings.cq = nullptr;
  }
  for (auto* umem : {&rings.wq_umem, &rings.dbr_umem, &rings.cq_umem}) {
    if (*umem != nullptr) {
      doca_umem_destroy(*umem);
      *umem = nullptr;
    }
  }
  for (auto* ring : {&rings.wq, &rings.dbr, &rings.cq_ring}) {
    if (*ring != nullptr) {
      doca_gpu_mem_free(gpu, *ring);
      *ring = nullptr;
    }
  }
  if (rings.uar != nullptr) {
    doca_uar_destroy(rings.uar);
    rings.uar = nullptr;
  }
}

// Registers GPU memory through a dma-buf, so that it works without the peermem module
struct ibv_mr* register_device_memory(struct ibv_pd* pd, void* base, size_t size, CUcontext owning,
                                      const std::string& what) {
  CudaContextRestoreGuard restore;
  if (!restore.valid() || (owning != nullptr && cuCtxSetCurrent(owning) != CUDA_SUCCESS)) {
    DAQIRI_LOG_CRITICAL("Could not activate the CUDA context of {}", what);
    return nullptr;
  }
  const size_t page = page_size();
  const auto va = reinterpret_cast<uintptr_t>(base);
  const uintptr_t aligned = va & ~(static_cast<uintptr_t>(page) - 1);
  const size_t offset = va - aligned;
  const size_t aligned_size = round_up(size + offset, page);
  int fd = -1;
  const CUresult res = cuMemGetHandleForAddressRange(
      &fd, static_cast<CUdeviceptr>(aligned), aligned_size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (res != CUDA_SUCCESS) {
    const char* es = nullptr;
    cuGetErrorString(res, &es);
    DAQIRI_LOG_CRITICAL(
        "cuMemGetHandleForAddressRange failed for {}: {}. CUDA DMA-BUF export requires Linux "
        "kernel 5.12 or newer and the NVIDIA open kernel driver",
        what, es != nullptr ? es : "?");
    return nullptr;
  }
  struct ibv_mr* mr = ibv_reg_dmabuf_mr(pd, offset, size, va, fd, IBV_ACCESS_LOCAL_WRITE);
  const int reg_errno = errno;
  close(fd);
  if (mr == nullptr) {
    DAQIRI_LOG_CRITICAL("ibv_reg_dmabuf_mr failed for {} ({} bytes): {}", what, size,
                        strerror(reg_errno));
  }
  return mr;
}

}  // namespace

// NIC port opened with libibverbs and DOCA Verbs, shared by the interfaces configured on it
struct GpunetioVerbsEngine::Device {
  std::string ibdev;
  std::string pci_addr;
  std::string netdev;
  struct ibv_context* ctx = nullptr;  // DevX context: the steering objects reference the queues
  struct ibv_pd* pd = nullptr;
  struct doca_verbs_context* vctx = nullptr;
  struct doca_verbs_pd* vpd = nullptr;
  struct doca_dev* dev = nullptr;
  int rx_port = -1;  // Interface with RX queues on this NIC
  std::map<std::string, struct ibv_mr*> mrs;

  // RX steering: transport domain of the TIRs, and the mlx5dv_dr rules
  struct mlx5dv_devx_obj* td = nullptr;
  uint32_t td_num = 0;
  struct mlx5dv_dr_domain* domain = nullptr;
  struct mlx5dv_dr_table* table = nullptr;
  std::vector<struct mlx5dv_dr_matcher*> matchers;
  std::vector<struct mlx5dv_dr_rule*> rules;
  std::vector<struct mlx5dv_dr_action*> tag_actions;
  std::vector<std::unique_ptr<Destination>> destinations;
};

// Where a steering rule sends packets: a TIR on one RX queue, or spreading them over several
struct GpunetioVerbsEngine::Destination {
  std::vector<int> queue_ids;
  struct mlx5dv_devx_obj* rqt = nullptr;
  struct mlx5dv_devx_obj* tir = nullptr;
  struct mlx5dv_dr_action* action = nullptr;
};

struct GpunetioVerbsEngine::RxQueue {
  // Configuration
  int port = 0;
  int id = 0;
  std::string name;
  std::string mr_name;
  int cpu_core = -1;
  uint32_t batch_size = 0;
  uint64_t timeout_ns = 0;
  Device* device = nullptr;

  // Receive ring: the first ring_size slots of stride bytes of the memory region, receive work
  // request i pointing at slot i
  uint8_t* base = nullptr;
  uint32_t stride = 0;
  uint64_t ring_size = 0;
  std::vector<void*> pkt_ptrs;
  struct doca_verbs_eth_rq* rq = nullptr;
  uint32_t rqn = 0;
  QueueRings rings;

  // Shared with the RX kernel
  gv::RxControl* ctrl = nullptr;
  gv::RxBurstDesc* desc = nullptr;  // ring_size entries
  uint32_t* pkt_len = nullptr;      // ring_size entries
  uint32_t* flow_tag = nullptr;     // ring_size entries
  gv::RxKernelArgs kernel_args{};   // The queue's block in the RX kernel

  // Burst bookkeeping. Descriptors are released in order: the worker advances the release
  // cursor over the descriptors whose packets the application freed.
  daqiri::Ring* ring = nullptr;  // Bursts ready for the application
  std::unique_ptr<uint32_t[]> desc_pkts;
  std::unique_ptr<std::atomic<uint32_t>[]> desc_freed;
  uint64_t consumed = 0;
  uint64_t release_cursor = 0;
  uint64_t released_pkts = 0;
  bool error_logged = false;

  std::atomic<uint64_t> bursts{0};
  std::atomic<uint64_t> pkts{0};
  std::atomic<uint64_t> bytes{0};
  std::atomic<uint64_t> pool_empty{0};
  std::atomic<uint64_t> ring_full{0};
};

struct GpunetioVerbsEngine::TxQueue {
  // Configuration
  int port = 0;
  int id = 0;
  std::string name;
  std::string mr_name;
  int cpu_core = -1;
  uint32_t batch_size = 0;
  Device* device = nullptr;

  // Send queue and the TX slots of the memory region
  struct doca_verbs_eth_sq* sq = nullptr;
  uint32_t sqn = 0;
  QueueRings rings;
  uint32_t lkey = 0;  // Network byte order
  uint8_t* slot_base = nullptr;
  size_t slot_size = 0;
  uint64_t num_slots = 0;
  uint32_t sq_wqes = 0;

  // Slots are cyclic: the application thread allocates at alloc_head. A burst's range of slots is
  // retired when the NIC completes it or when the application frees it unsent, in any order;
  // completed_tail covers the slots before the first range still in use.
  uint64_t alloc_head = 0;
  std::atomic<uint64_t> completed_tail{0};
  std::mutex retire_mutex;
  std::map<uint64_t, uint64_t> retired;  // First slot to count, for ranges past completed_tail

  void retire(uint64_t first, uint64_t count) {
    if (count == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(retire_mutex);
    uint64_t tail = completed_tail.load(std::memory_order_relaxed);
    retired[first] = count;
    for (auto it = retired.begin(); it != retired.end() && it->first == tail;
         it = retired.erase(it)) {
      tail += it->second;
    }
    completed_tail.store(tail, std::memory_order_release);
  }

  // Shared with the TX kernel
  gv::TxPacket* pkts = nullptr;  // num_slots entries
  gv::TxBurstDesc* desc = nullptr;
  gv::TxControl* ctrl = nullptr;
  gv::TxKernelArgs kernel_args{};  // The queue's block in the TX kernel

  daqiri::Ring* send_ring = nullptr;  // Bursts sent by the application
  std::deque<BurstParams*> inflight;
  uint64_t published = 0;
  uint64_t reaped = 0;
  std::atomic<bool> failed{false};

  std::atomic<uint64_t> sent_bursts{0};
  std::atomic<uint64_t> sent_pkts{0};
  std::atomic<uint64_t> sent_bytes{0};
  std::atomic<uint64_t> send_ring_full{0};
};

GpunetioVerbsEngine::GpunetioVerbsEngine() = default;

GpunetioVerbsEngine::~GpunetioVerbsEngine() {
  shutdown();
}

// ---------------------------------------------------------------------------
// Bring-up
// ---------------------------------------------------------------------------
bool GpunetioVerbsEngine::set_config_and_initialize(const NetworkConfig& cfg) {
  if (initialized_) {
    DAQIRI_LOG_ERROR("gpunetio_verbs engine is already initialized; call shutdown() first");
    return false;
  }

  cfg_ = cfg;
  if (!validate_config()) {
    DAQIRI_LOG_CRITICAL("Config validation failed");
    return false;
  }
  initialize();
  if (!initialized_) {
    shutdown();
  }
  return initialized_;
}

void GpunetioVerbsEngine::initialize() {
  DAQIRI_LOG_INFO("Initializing gpunetio_verbs raw engine (experimental)");
  set_doca_sdk_log_level(cfg_.log_level_);

  int if_num = 0;
  for (auto& intf : cfg_.ifs_) {
    intf.port_id_ = if_num++;
  }

  for (const auto& intf : cfg_.ifs_) {
    for (const auto& cfg_q : intf.rx_.queues_) {
      auto q = std::make_unique<RxQueue>();
      q->port = intf.port_id_;
      q->id = cfg_q.common_.id_;
      q->name = cfg_q.common_.name_;
      q->mr_name = cfg_q.common_.mrs_[0];
      q->cpu_core = parse_core(cfg_q.common_.cpu_core_);
      q->batch_size = static_cast<uint32_t>(std::max(cfg_q.common_.batch_size_, 1));
      q->timeout_ns = cfg_q.timeout_us_ * 1000;
      rx_queues_.push_back(std::move(q));
    }
    for (const auto& cfg_q : intf.tx_.queues_) {
      auto q = std::make_unique<TxQueue>();
      q->port = intf.port_id_;
      q->id = cfg_q.common_.id_;
      q->name = cfg_q.common_.name_;
      q->mr_name = cfg_q.common_.mrs_[0];
      q->cpu_core = parse_core(cfg_q.common_.cpu_core_);
      q->batch_size = static_cast<uint32_t>(std::max(cfg_q.common_.batch_size_, 1));
      tx_queues_.push_back(std::move(q));
    }
  }

  if (!setup_gpu() || !open_devices() || !size_memory_regions()) {
    return;
  }
  if (allocate_memory_regions() != Status::SUCCESS) {
    DAQIRI_LOG_CRITICAL("Failed to allocate memory regions");
    return;
  }
  if (!register_memory_regions()) {
    return;
  }
  for (auto& q : rx_queues_) {
    if (!create_rx_queue(*q)) {
      return;
    }
  }
  for (auto& q : tx_queues_) {
    if (!create_tx_queue(*q)) {
      return;
    }
  }
  for (auto& dev : devices_) {
    if (dev->rx_port >= 0 && !program_steering(*dev, cfg_.ifs_[dev->rx_port])) {
      return;
    }
  }
  if (!create_burst_pools() || !start_kernels()) {
    return;
  }

  initialized_ = true;
  DAQIRI_LOG_INFO("gpunetio_verbs engine initialized with {} RX queue(s), {} TX queue(s)",
                  rx_queues_.size(), tx_queues_.size());

  // Engines self-start their workers at the end of initialize()
  run();
}

// The kernels and the queue rings live on one GPU: the one the memory regions of the queues name
// in their affinity
bool GpunetioVerbsEngine::setup_gpu() {
  gpu_ordinal_ = 0;
  bool rx_device_memory = false;
  for (const auto& intf : cfg_.ifs_) {
    for (const auto& q : intf.rx_.queues_) {
      const auto& mr = cfg_.mrs_.at(q.common_.mrs_[0]);
      gpu_ordinal_ = mr.affinity_;
      rx_device_memory = rx_device_memory || mr.kind_ == MemoryKind::DEVICE;
    }
    for (const auto& q : intf.tx_.queues_) {
      gpu_ordinal_ = cfg_.mrs_.at(q.common_.mrs_[0]).affinity_;
    }
  }

  if (!select_cuda_device(gpu_ordinal_, "initializing the gpunetio_verbs engine")) {
    return false;
  }
  // GPUNetIO needs a CUDA context on the device
  cudaError_t err = cudaFree(nullptr);
  if (err != cudaSuccess) {
    DAQIRI_LOG_CRITICAL("CUDA initialization failed: {}", cudaGetErrorString(err));
    return false;
  }
  char bus_id[32] = {};
  err = cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), gpu_ordinal_);
  if (err != cudaSuccess) {
    DAQIRI_LOG_CRITICAL("Could not get the PCIe address of CUDA device {}: {}", gpu_ordinal_,
                        cudaGetErrorString(err));
    return false;
  }
  for (char* c = bus_id; *c != '\0'; c++) {
    *c = static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
  }

  // The RX kernel reads only the completions. The application reads the packets once the CPU
  // hands it their burst: on GPUs that don't order GPUDirect RDMA writes for their own kernels,
  // the RX workers flush the writes first.
  if (rx_device_memory) {
    CUdevice cu_dev = 0;
    int ordering = 0;
    int flush_options = 0;
    if (cuDeviceGet(&cu_dev, gpu_ordinal_) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&ordering, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                             cu_dev) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&flush_options,
                             CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,
                             cu_dev) != CUDA_SUCCESS) {
      DAQIRI_LOG_CRITICAL("Could not query the GPUDirect RDMA ordering of CUDA device {}",
                          gpu_ordinal_);
      return false;
    }
    flush_rx_writes_ = ordering < CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER;
    if (flush_rx_writes_ && (flush_options & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST) == 0) {
      DAQIRI_LOG_CRITICAL(
          "CUDA device {} neither orders nor flushes GPUDirect RDMA writes: RX queues need "
          "'host_pinned' memory regions",
          gpu_ordinal_);
      return false;
    }
  }

  const doca_error_t ret = doca_gpu_create(bus_id, &gpu_);
  if (ret != DOCA_SUCCESS) {
    gpu_ = nullptr;
    DAQIRI_LOG_CRITICAL("doca_gpu_create({}) failed: {}", bus_id, doca_err(ret));
    return false;
  }
  DAQIRI_LOG_INFO("CUDA device {} ({}) runs the gpunetio_verbs datapath{}", gpu_ordinal_, bus_id,
                  flush_rx_writes_ ? ", flushing GPUDirect RDMA writes before each RX burst" : "");
  return true;
}

// Resolves every interface address, then name, to a NIC port with DOCA, and opens it with
// libibverbs (DevX) and DOCA Verbs. Interfaces on the same port share one device; only one of
// them may have RX queues, as the steering rules belong to the port.
bool GpunetioVerbsEngine::open_devices() {
  struct doca_devinfo** list = nullptr;
  uint32_t num = 0;
  doca_error_t ret = doca_devinfo_create_list(&list, &num);
  if (ret != DOCA_SUCCESS) {
    DAQIRI_LOG_CRITICAL("Failed to list DOCA devices: {}", doca_err(ret));
    return false;
  }
  int num_ibv = 0;
  struct ibv_device** ibv_list = ibv_get_device_list(&num_ibv);
  if (ibv_list == nullptr) {
    doca_devinfo_destroy_list(list);
    DAQIRI_LOG_CRITICAL("No ibverbs devices found: {}", strerror(errno));
    return false;
  }

  bool ok = true;
  port_devices_.assign(cfg_.ifs_.size(), nullptr);
  for (const auto& intf : cfg_.ifs_) {
    struct doca_devinfo* match = nullptr;
    for (const std::string* key : {&intf.address_, &intf.name_}) {
      for (uint32_t i = 0; i < num && match == nullptr; i++) {
        if (devinfo_matches(list[i], *key)) {
          match = list[i];
        }
      }
      if (match != nullptr) {
        break;
      }
    }
    if (match == nullptr) {
      DAQIRI_LOG_CRITICAL(
          "Interface '{}' (address '{}') matches no NIC; use a PCIe address, an IB device name or "
          "a netdev name",
          intf.name_, intf.address_);
      ok = false;
      break;
    }

    char ibdev[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {};
    (void)doca_devinfo_get_ibdev_name(match, ibdev, sizeof(ibdev));
    Device* device = nullptr;
    for (auto& dev : devices_) {
      if (dev->ibdev == ibdev) {
        device = dev.get();
      }
    }
    if (device == nullptr) {
      auto dev = std::make_unique<Device>();
      dev->ibdev = ibdev;
      char pci_addr[DOCA_DEVINFO_PCI_ADDR_SIZE] = {};
      if (doca_devinfo_get_pci_addr_str(match, pci_addr) == DOCA_SUCCESS) {
        dev->pci_addr = pci_addr;
      }
      char netdev[DOCA_DEVINFO_IFACE_NAME_SIZE] = {};
      if (doca_devinfo_get_iface_name(match, netdev, sizeof(netdev)) == DOCA_SUCCESS) {
        dev->netdev = netdev;
      }
      Device& d = *dev;
      devices_.push_back(std::move(dev));

      struct ibv_device* ibv_dev = nullptr;
      for (int i = 0; i < num_ibv; i++) {
        if (d.ibdev == ibv_get_device_name(ibv_list[i])) {
          ibv_dev = ibv_list[i];
        }
      }
      if (ibv_dev == nullptr || !mlx5dv_is_supported(ibv_dev)) {
        DAQIRI_LOG_CRITICAL("{} is not an mlx5 ibverbs device", d.ibdev);
        ok = false;
        break;
      }
      // DevX: the steering objects reference the DOCA Verbs queues, which DOCA creates on this
      // context
      struct mlx5dv_context_attr dv_attr {};
      dv_attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
      d.ctx = mlx5dv_open_device(ibv_dev, &dv_attr);
      if (d.ctx == nullptr) {
        DAQIRI_LOG_CRITICAL("mlx5dv_open_device(DEVX) failed for {}: {}", d.ibdev, strerror(errno));
        ok = false;
        break;
      }
      d.pd = ibv_alloc_pd(d.ctx);
      if (d.pd == nullptr) {
        DAQIRI_LOG_CRITICAL("ibv_alloc_pd({}) failed: {}", d.ibdev, strerror(errno));
        ok = false;
        break;
      }
      ret = doca_verbs_bridge_verbs_context_import(d.ctx, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE,
                                                   &d.vctx);
      if (ret == DOCA_SUCCESS) {
        ret = doca_verbs_bridge_verbs_pd_import(d.pd, &d.vpd);
      }
      if (ret == DOCA_SUCCESS) {
        ret = doca_rdma_bridge_open_dev_from_pd(d.pd, &d.dev);
      }
      if (ret != DOCA_SUCCESS) {
        DAQIRI_LOG_CRITICAL("Could not open {} with DOCA Verbs: {}", d.ibdev, doca_err(ret));
        ok = false;
        break;
      }
      ret = doca_gpu_nic_cap_is_nic_handler_gpu_sm_db_supported(gpu_, d.dev);
      if (ret != DOCA_SUCCESS) {
        DAQIRI_LOG_CRITICAL(
            "CUDA device {} can't ring the doorbells of {}: {}. The gpunetio_verbs engine needs "
            "the GPU to drive the NIC directly (GPU_SM_DB); check that the GPU and the NIC can "
            "use GPUDirect RDMA and that the NVIDIA driver maps the NIC doorbells",
            gpu_ordinal_, d.ibdev, doca_err(ret));
        ok = false;
        break;
      }
      device = &d;
    }
    port_devices_[intf.port_id_] = device;

    if (!intf.rx_.queues_.empty()) {
      if (device->rx_port >= 0) {
        DAQIRI_LOG_CRITICAL(
            "Interfaces '{}' and '{}' both define RX queues on {}; only one interface per NIC "
            "port may",
            cfg_.ifs_[device->rx_port].name_, intf.name_, device->ibdev);
        ok = false;
        break;
      }
      device->rx_port = intf.port_id_;
    }
    DAQIRI_LOG_INFO("Interface '{}' uses {} ({}, {})", intf.name_, device->ibdev, device->pci_addr,
                    device->netdev);
  }
  ibv_free_device_list(ibv_list);
  doca_devinfo_destroy_list(list);
  if (!ok) {
    return false;
  }

  for (auto& q : rx_queues_) {
    q->device = port_devices_[q->port];
  }
  for (auto& q : tx_queues_) {
    q->device = port_devices_[q->port];
  }
  return true;
}

// Slots of stride bytes. The region of an RX queue holds its receive ring, a power-of-two count of
// slots: DAQIRI rounds the count of a region it allocates up, and uses the first slots of a
// caller-owned region.
bool GpunetioVerbsEngine::size_memory_regions() {
  for (auto& [name, mr] : cfg_.mrs_) {
    (void)name;
    mr.adj_size_ = round_up(std::max<size_t>(mr.buf_size_, SLOT_ALIGN), SLOT_ALIGN);
  }

  for (auto& q : rx_queues_) {
    auto& mr = cfg_.mrs_.at(q->mr_name);
    q->stride = static_cast<uint32_t>(mr.adj_size_);
    // A region bound by the application keeps the layout it was sized with
    if (external_mrs_.count(q->mr_name) == 0) {
      q->ring_size = next_pow2(std::max<uint64_t>(
          {static_cast<uint64_t>(mr.num_bufs_), RX_RING_MIN_PKTS, q->batch_size}));
      if (q->ring_size > RX_RING_MAX_PKTS) {
        q->ring_size = RX_RING_MAX_PKTS;
      }
      if (q->ring_size != mr.num_bufs_) {
        DAQIRI_LOG_INFO(
            "RX queue '{}': memory region '{}' holds a ring of {} slots ({} configured)", q->name,
            q->mr_name, q->ring_size, mr.num_bufs_);
      }
      mr.num_bufs_ = q->ring_size;
    } else {
      q->ring_size = prev_pow2(std::min<uint64_t>(mr.num_bufs_, RX_RING_MAX_PKTS));
      if (q->ring_size < std::max<uint64_t>(RX_RING_MIN_PKTS, q->batch_size)) {
        DAQIRI_LOG_CRITICAL(
            "RX queue '{}': caller-owned memory region '{}' needs at least {} buffers", q->name,
            q->mr_name, std::max<uint64_t>(RX_RING_MIN_PKTS, q->batch_size));
        return false;
      }
    }
    if (q->batch_size > q->ring_size) {
      DAQIRI_LOG_CRITICAL("RX queue '{}': batch_size {} exceeds the ring of {} slots", q->name,
                          q->batch_size, q->ring_size);
      return false;
    }

    if (!q->device->netdev.empty()) {
      unsigned mtu = 0;
      const std::string path = "/sys/class/net/" + q->device->netdev + "/mtu";
      if (FILE* f = std::fopen(path.c_str(), "r")) {
        if (std::fscanf(f, "%u", &mtu) != 1) {
          mtu = 0;
        }
        std::fclose(f);
      }
      // Ethernet header and one VLAN tag
      if (mtu != 0 && mtu + 18 > q->stride) {
        DAQIRI_LOG_WARN(
            "RX queue '{}': {} has MTU {}, but a packet bigger than the {} byte ring slot stops "
            "the queue; raise buf_size or lower the MTU",
            q->name, q->device->netdev, mtu, q->stride);
      }
    }
  }
  return true;
}

// Registers the memory region of each queue with the NIC of the queue
bool GpunetioVerbsEngine::register_memory_regions() {
  const auto add = [this](const std::string& mr_name, Device* device) {
    if (device->mrs.count(mr_name) != 0) {
      return true;
    }
    const auto& mr = cfg_.mrs_.at(mr_name);
    const auto& region = ar_.at(mr_name);
    struct ibv_mr* reg = nullptr;
    if (mr.kind_ == MemoryKind::DEVICE) {
      bool dmabuf = false;
      if (!get_cuda_dmabuf_support(mr.affinity_, &dmabuf)) {
        return false;
      }
      if (!dmabuf) {
        log_cuda_dmabuf_unavailable(mr);
        return false;
      }
      reg = register_device_memory(device->pd, region.ptr_, mr.ttl_size_, region.cuda_context_,
                                   "memory region " + mr_name);
    } else {
      reg = ibv_reg_mr(device->pd, region.ptr_, mr.ttl_size_, IBV_ACCESS_LOCAL_WRITE);
      if (reg == nullptr) {
        DAQIRI_LOG_CRITICAL("ibv_reg_mr failed for memory region {} ({} bytes): {}", mr_name,
                            mr.ttl_size_, strerror(errno));
      }
    }
    if (reg == nullptr) {
      return false;
    }
    device->mrs[mr_name] = reg;
    DAQIRI_LOG_INFO("Memory region '{}' registered on {} ({} bytes, lkey {:#x})", mr_name,
                    device->ibdev, mr.ttl_size_, reg->lkey);
    return true;
  };
  for (auto& q : rx_queues_) {
    if (!add(q->mr_name, q->device)) {
      return false;
    }
  }
  for (auto& q : tx_queues_) {
    if (!add(q->mr_name, q->device)) {
      return false;
    }
  }
  return true;
}

// An Ethernet receive queue whose work requests point at the ring slots, exported to the GPU. DOCA
// GPUNetIO exports send and receive queues in pairs: the send side of an RX queue's handle points
// at its own receive ring and at the unused send counter of its doorbell record, and the kernel
// never uses it.
bool GpunetioVerbsEngine::create_rx_queue(RxQueue& q) {
  if (!select_cuda_device(gpu_ordinal_, "creating RX queue '" + q.name + "'")) {
    return false;
  }
  Device& d = *q.device;
  const std::string name = "RX queue '" + q.name + "'";
  q.base = static_cast<uint8_t*>(ar_.at(q.mr_name).ptr_);
  const uint32_t lkey = d.mrs.at(q.mr_name)->lkey;

  if (!create_cq(gpu_, d.dev, d.vctx, static_cast<uint32_t>(q.ring_size), q.rings, name) ||
      !alloc_work_queue(gpu_, d.dev, std::max<size_t>(q.ring_size * RQ_WQE_SIZE, 64), q.rings,
                        name) ||
      !create_uar(d.dev, q.rings, name)) {
    return false;
  }

  struct doca_verbs_eth_rq_init_attr* attr = nullptr;
  doca_error_t ret = doca_verbs_eth_rq_init_attr_create(&attr);
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_rq_init_attr_set_pd(attr, d.vpd);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_rq_init_attr_set_cq(attr, q.rings.cq);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_rq_init_attr_set_wr_num(attr, static_cast<uint32_t>(q.ring_size));
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_rq_init_attr_set_max_sges(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_rq_init_attr_set_external_datapath_en(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_rq_init_attr_set_external_umem(attr, q.rings.wq_umem, 0);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_rq_init_attr_set_external_dbr_umem(attr, q.rings.dbr_umem, 0);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_rq_create(d.vctx, attr, &q.rq);
  }
  if (attr != nullptr) {
    doca_verbs_eth_rq_init_attr_destroy(attr);
  }
  if (ret != DOCA_SUCCESS) {
    q.rq = nullptr;
    DAQIRI_LOG_CRITICAL("Could not create the Ethernet receive queue of {}: {}", name,
                        doca_err(ret));
    return false;
  }
  q.rqn = doca_verbs_eth_rq_get_rq_hw_number(q.rq);

  void* wq = nullptr;
  uint32_t wqes = 0;
  uint32_t wqe_size = 0;
  ret = doca_verbs_eth_rq_get_wq(q.rq, &wq, &wqes, &wqe_size);
  if (ret != DOCA_SUCCESS || wq != q.rings.wq || wqes != q.ring_size || wqe_size != RQ_WQE_SIZE) {
    DAQIRI_LOG_CRITICAL(
        "{}: unexpected receive queue layout ({} work requests of {} bytes, expected {} of {})",
        name, wqes, wqe_size, q.ring_size, RQ_WQE_SIZE);
    return false;
  }
  // Receive work request i points at ring slot i for good
  std::vector<struct mlx5_wqe_data_seg> rwqes(q.ring_size);
  q.pkt_ptrs.resize(q.ring_size);
  for (uint64_t i = 0; i < q.ring_size; i++) {
    q.pkt_ptrs[i] = q.base + i * q.stride;
    rwqes[i].byte_count = htobe32(q.stride);
    rwqes[i].lkey = htobe32(lkey);
    rwqes[i].addr = htobe64(reinterpret_cast<uint64_t>(q.pkt_ptrs[i]));
  }
  if (cudaMemcpy(wq, rwqes.data(), rwqes.size() * sizeof(rwqes[0]), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    DAQIRI_LOG_CRITICAL("{}: could not write the receive work requests", name);
    return false;
  }

  void* uar_reg = nullptr;
  ret = doca_uar_reg_addr_get(q.rings.uar, &uar_reg);
  const CqLayout cq = cq_layout(q.rings.cq);
  auto* dbr = static_cast<uint32_t*>(q.rings.dbr);
  if (ret == DOCA_SUCCESS) {
    ret = doca_gpu_verbs_bridge_export_qp(
        gpu_, q.rqn, wq, 1, dbr + DOCA_GPUNETIO_MLX5_SND_DBR, static_cast<uint64_t*>(uar_reg),
        UAR_DB_SIZE, 0, nullptr, 0, nullptr, q.rqn, wq, static_cast<uint16_t>(q.ring_size),
        dbr + DOCA_GPUNETIO_MLX5_RCV_DBR, RQ_WQE_SIZE, doca_verbs_cq_get_cqn(q.rings.cq), cq.cqes,
        cq.entries, cq.dbr, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, &q.rings.gpu_qp);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_gpu_verbs_get_qp_dev(q.rings.gpu_qp, &q.rings.dev_qp);
  }
  if (ret != DOCA_SUCCESS) {
    DAQIRI_LOG_CRITICAL("Could not export {} to the GPU: {}", name, doca_err(ret));
    return false;
  }

  q.ctrl = alloc_pinned<gv::RxControl>(1);
  q.desc = alloc_pinned<gv::RxBurstDesc>(q.ring_size);
  q.pkt_len = alloc_pinned<uint32_t>(q.ring_size);
  q.flow_tag = alloc_pinned<uint32_t>(q.ring_size);
  if (q.ctrl == nullptr || q.desc == nullptr || q.pkt_len == nullptr || q.flow_tag == nullptr) {
    DAQIRI_LOG_CRITICAL("{}: could not allocate the kernel resources", name);
    return false;
  }
  const uint32_t threads = gv::rx_kernel_threads();
  gv::RxKernelArgs& args = q.kernel_args;
  args.qp = q.rings.dev_qp;
  args.desc = q.desc;
  args.pkt_len = q.pkt_len;
  args.flow_tag = q.flow_tag;
  args.ctrl = q.ctrl;
  args.timeout_ns = q.timeout_ns;
  args.ring_mask = static_cast<uint32_t>(q.ring_size - 1);
  args.desc_mask = static_cast<uint32_t>(q.ring_size - 1);
  args.batch_size = q.batch_size;
  args.max_pkts = static_cast<uint32_t>(
      round_up(std::min<uint64_t>({q.batch_size, RX_MAX_PKTS_PER_ROUND, q.ring_size}), threads));

  // At most ring_size descriptors are outstanding, as each holds at least one packet
  q.ring =
      daqiri::Ring::create(("GV_RX_" + std::to_string(q.port) + "_" + std::to_string(q.id)).c_str(),
                           static_cast<unsigned>(q.ring_size) + 1, daqiri::RingMode::MPMC);
  q.desc_pkts.reset(new (std::nothrow) uint32_t[q.ring_size]());
  q.desc_freed.reset(new (std::nothrow) std::atomic<uint32_t>[q.ring_size]);
  if (q.ring == nullptr || !q.desc_pkts || !q.desc_freed) {
    DAQIRI_LOG_CRITICAL("{}: could not allocate the burst bookkeeping", name);
    return false;
  }
  for (uint64_t i = 0; i < q.ring_size; i++) {
    q.desc_freed[i].store(0, std::memory_order_relaxed);
  }

  DAQIRI_LOG_INFO("{} (port {} queue {}): receive queue {:#x}, ring of {} slots of {} bytes", name,
                  q.port, q.id, q.rqn, q.ring_size, q.stride);
  return true;
}

// An Ethernet send queue exported to the GPU. The receive side of its handle points at the unused
// receive counter of its doorbell record, and the kernel never uses it.
bool GpunetioVerbsEngine::create_tx_queue(TxQueue& q) {
  if (!select_cuda_device(gpu_ordinal_, "creating TX queue '" + q.name + "'")) {
    return false;
  }
  Device& d = *q.device;
  const std::string name = "TX queue '" + q.name + "'";
  const auto& mr = cfg_.mrs_.at(q.mr_name);
  q.slot_base = static_cast<uint8_t*>(ar_.at(q.mr_name).ptr_);
  q.slot_size = mr.adj_size_;
  q.num_slots = mr.num_bufs_;
  q.lkey = htobe32(d.mrs.at(q.mr_name)->lkey);
  q.sq_wqes = static_cast<uint32_t>(
      next_pow2(std::clamp<uint64_t>(q.num_slots, TX_SQ_MIN_WQES, TX_SQ_MAX_WQES)));

  // One work request of one WQEBB per packet. The completion queue has the size of the send queue,
  // as DOCA GPUNetIO expects, although a completion closes each chunk of work requests.
  if (!create_cq(gpu_, d.dev, d.vctx, q.sq_wqes, q.rings, name) ||
      !alloc_work_queue(gpu_, d.dev, static_cast<size_t>(q.sq_wqes) * SQ_WQE_SIZE, q.rings, name) ||
      !create_uar(d.dev, q.rings, name)) {
    return false;
  }

  struct doca_verbs_eth_sq_init_attr* attr = nullptr;
  doca_error_t ret = doca_verbs_eth_sq_init_attr_create(&attr);
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_sq_init_attr_set_pd(attr, d.vpd);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_sq_init_attr_set_cq(attr, q.rings.cq);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_sq_init_attr_set_wr_num(attr, q.sq_wqes);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_sq_init_attr_set_max_sges(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_sq_init_attr_set_external_datapath_en(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_sq_init_attr_set_external_umem(attr, q.rings.wq_umem, 0);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_sq_init_attr_set_external_dbr_umem(attr, q.rings.dbr_umem, 0);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_sq_init_attr_set_external_uar(attr, q.rings.uar);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_eth_sq_create(d.vctx, attr, &q.sq);
  }
  if (attr != nullptr) {
    doca_verbs_eth_sq_init_attr_destroy(attr);
  }
  if (ret != DOCA_SUCCESS) {
    q.sq = nullptr;
    DAQIRI_LOG_CRITICAL("Could not create the Ethernet send queue of {}: {}", name, doca_err(ret));
    return false;
  }
  q.sqn = doca_verbs_eth_sq_get_sq_hw_number(q.sq);

  void* wq = nullptr;
  uint32_t wqebbs = 0;
  ret = doca_verbs_eth_sq_get_wq(q.sq, &wq, &wqebbs);
  if (ret != DOCA_SUCCESS || wq != q.rings.wq || wqebbs != q.sq_wqes) {
    DAQIRI_LOG_CRITICAL("{}: unexpected send queue layout ({} WQEBBs, expected {})", name, wqebbs,
                        q.sq_wqes);
    return false;
  }

  const CqLayout cq = cq_layout(q.rings.cq);
  auto* dbr = static_cast<uint32_t*>(q.rings.dbr);
  ret = doca_gpu_verbs_bridge_export_qp(
      gpu_, q.sqn, wq, static_cast<uint16_t>(q.sq_wqes), dbr + DOCA_GPUNETIO_MLX5_SND_DBR,
      static_cast<uint64_t*>(doca_verbs_eth_sq_get_uar_addr(q.sq)), UAR_DB_SIZE,
      doca_verbs_cq_get_cqn(q.rings.cq), cq.cqes, cq.entries, cq.dbr, q.sqn, wq, 1,
      dbr + DOCA_GPUNETIO_MLX5_RCV_DBR, RQ_WQE_SIZE, 0, nullptr, 0, nullptr,
      DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, &q.rings.gpu_qp);
  if (ret == DOCA_SUCCESS) {
    ret = doca_gpu_verbs_get_qp_dev(q.rings.gpu_qp, &q.rings.dev_qp);
  }
  if (ret != DOCA_SUCCESS) {
    DAQIRI_LOG_CRITICAL("Could not export {} to the GPU: {}", name, doca_err(ret));
    return false;
  }

  q.pkts = alloc_pinned<gv::TxPacket>(q.num_slots);
  q.desc = alloc_pinned<gv::TxBurstDesc>(TX_DESC_RING);
  q.ctrl = alloc_pinned<gv::TxControl>(1);
  if (q.pkts == nullptr || q.desc == nullptr || q.ctrl == nullptr) {
    DAQIRI_LOG_CRITICAL("{}: could not allocate the kernel resources", name);
    return false;
  }
  gv::TxKernelArgs& args = q.kernel_args;
  args.qp = q.rings.dev_qp;
  args.pkts = q.pkts;
  args.desc = q.desc;
  args.ctrl = q.ctrl;
  args.num_slots = q.num_slots;
  args.desc_mask = TX_DESC_RING - 1;
  args.lkey = q.lkey;
  args.max_chunk = std::max<uint32_t>(1, q.sq_wqes / TX_CHUNKS_PER_SQ);
  q.send_ring =
      daqiri::Ring::create(("GV_TX_" + std::to_string(q.port) + "_" + std::to_string(q.id)).c_str(),
                           TX_SEND_RING, daqiri::RingMode::MPMC);
  if (q.send_ring == nullptr) {
    DAQIRI_LOG_CRITICAL("{}: could not allocate the send ring", name);
    return false;
  }

  DAQIRI_LOG_INFO("{} (port {} queue {}): send queue {:#x} of {} WQEBBs, {} slots of {} bytes",
                  name, q.port, q.id, q.sqn, q.sq_wqes, q.num_slots, q.slot_size);
  return true;
}

// ---------------------------------------------------------------------------
// RX steering
// ---------------------------------------------------------------------------

// The TIR of a set of RX queues: a direct TIR for one queue, a Toeplitz hash over the IPv4
// addresses and UDP ports of the packets for several
GpunetioVerbsEngine::Destination* GpunetioVerbsEngine::rx_destination(
    Device& dev, const std::vector<RxQueue*>& queues) {
  std::vector<int> ids;
  for (const RxQueue* q : queues) {
    ids.push_back(q->id);
  }
  for (auto& destination : dev.destinations) {
    if (destination->queue_ids == ids) {
      return destination.get();
    }
  }
  auto destination = std::make_unique<Destination>();
  destination->queue_ids = ids;
  Destination& dst = *destination;
  dev.destinations.push_back(std::move(destination));

  uint32_t tir_in[DEVX_ST_SZ_DW(create_tir_in)] = {};
  uint32_t tir_out[DEVX_ST_SZ_DW(create_tir_out)] = {};
  void* tirc = DEVX_ADDR_OF(create_tir_in, tir_in, ctx);
  DEVX_SET(create_tir_in, tir_in, opcode, MLX5_CMD_OP_CREATE_TIR);
  DEVX_SET(tirc, tirc, transport_domain, dev.td_num);
  if (queues.size() == 1) {
    DEVX_SET(tirc, tirc, disp_type, MLX5_TIRC_DISP_TYPE_DIRECT);
    DEVX_SET(tirc, tirc, inline_rqn, queues.front()->rqn);
    DEVX_SET(tirc, tirc, rx_hash_fn, MLX5_RX_HASH_FN_NONE);
  } else {
    const size_t table_size = rss_table_size(queues.size());
    const size_t rqt_bytes = DEVX_ST_SZ_BYTES(create_rqt_in) + (table_size - 1) * sizeof(uint32_t);
    std::vector<uint32_t> rqt_in((rqt_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t), 0);
    uint32_t rqt_out[DEVX_ST_SZ_DW(create_rqt_out)] = {};
    void* rqtc = DEVX_ADDR_OF(create_rqt_in, rqt_in.data(), rqt_context);
    DEVX_SET(create_rqt_in, rqt_in.data(), opcode, MLX5_CMD_OP_CREATE_RQT);
    DEVX_SET(rqtc, rqtc, rqt_max_size, table_size);
    DEVX_SET(rqtc, rqtc, rqt_actual_size, table_size);
    auto* entries = reinterpret_cast<uint32_t*>(DEVX_ADDR_OF(rqtc, rqtc, rq_num));
    for (size_t i = 0; i < table_size; i++) {
      entries[i] = htobe32(queues[i % queues.size()]->rqn & 0x00ffffffU);
    }
    dst.rqt = mlx5dv_devx_obj_create(dev.ctx, rqt_in.data(), rqt_bytes, rqt_out, sizeof(rqt_out));
    if (dst.rqt == nullptr) {
      DAQIRI_LOG_CRITICAL("CREATE_RQT failed on {}: {} (syndrome {:#x})", dev.ibdev,
                          strerror(errno), DEVX_GET(create_rqt_out, rqt_out, syndrome));
      return nullptr;
    }
    DEVX_SET(tirc, tirc, disp_type, MLX5_TIRC_DISP_TYPE_INDIRECT);
    DEVX_SET(tirc, tirc, indirect_table, DEVX_GET(create_rqt_out, rqt_out, rqtn));
    DEVX_SET(tirc, tirc, rx_hash_fn, MLX5_RX_HASH_FN_TOEPLITZ);
    void* selector = DEVX_ADDR_OF(tirc, tirc, rx_hash_field_selector_outer);
    DEVX_SET(rx_hash_field_select, selector, l3_prot_type, MLX5_L3_PROT_TYPE_IPV4);
    DEVX_SET(rx_hash_field_select, selector, l4_prot_type, MLX5_L4_PROT_TYPE_UDP);
    DEVX_SET(rx_hash_field_select, selector, selected_fields,
             MLX5_HASH_FIELD_SEL_SRC_IP | MLX5_HASH_FIELD_SEL_DST_IP |
                 MLX5_HASH_FIELD_SEL_L4_SPORT | MLX5_HASH_FIELD_SEL_L4_DPORT);
    std::memcpy(DEVX_ADDR_OF(tirc, tirc, rx_hash_toeplitz_key), kToeplitzRssKey.data(),
                kToeplitzRssKey.size());
  }
  dst.tir = mlx5dv_devx_obj_create(dev.ctx, tir_in, sizeof(tir_in), tir_out, sizeof(tir_out));
  if (dst.tir == nullptr) {
    DAQIRI_LOG_CRITICAL("CREATE_TIR failed on {}: {} (syndrome {:#x})", dev.ibdev, strerror(errno),
                        DEVX_GET(create_tir_out, tir_out, syndrome));
    return nullptr;
  }
  dst.action = mlx5dv_dr_action_create_dest_devx_tir(dst.tir);
  if (dst.action == nullptr) {
    DAQIRI_LOG_CRITICAL("Could not create a TIR destination action on {}: {}", dev.ibdev,
                        strerror(errno));
    return nullptr;
  }
  return &dst;
}

// mlx5dv_dr rules in the root table of the port's receive domain match the configured flows in
// order. Packets no rule matches go on to the kernel with flow_isolation, and otherwise to the
// first RX queue.
bool GpunetioVerbsEngine::program_steering(Device& dev, const InterfaceConfig& intf) {
  uint32_t td_in[DEVX_ST_SZ_DW(alloc_transport_domain_in)] = {};
  uint32_t td_out[DEVX_ST_SZ_DW(alloc_transport_domain_out)] = {};
  DEVX_SET(alloc_transport_domain_in, td_in, opcode, MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN);
  dev.td = mlx5dv_devx_obj_create(dev.ctx, td_in, sizeof(td_in), td_out, sizeof(td_out));
  if (dev.td == nullptr) {
    DAQIRI_LOG_CRITICAL(
        "ALLOC_TRANSPORT_DOMAIN failed on {}: {} (syndrome {:#x}). Steering raw Ethernet needs "
        "root or CAP_NET_RAW",
        dev.ibdev, strerror(errno), DEVX_GET(alloc_transport_domain_out, td_out, syndrome));
    return false;
  }
  dev.td_num = DEVX_GET(alloc_transport_domain_out, td_out, transport_domain);
  dev.domain = mlx5dv_dr_domain_create(dev.ctx, MLX5DV_DR_DOMAIN_TYPE_NIC_RX);
  if (dev.domain == nullptr) {
    DAQIRI_LOG_CRITICAL("mlx5dv_dr_domain_create failed on {}: {}", dev.ibdev, strerror(errno));
    return false;
  }
  dev.table = mlx5dv_dr_table_create(dev.domain, 0);
  if (dev.table == nullptr) {
    DAQIRI_LOG_CRITICAL("mlx5dv_dr_table_create failed on {}: {}", dev.ibdev, strerror(errno));
    return false;
  }

  std::map<int, RxQueue*> queues;
  for (auto& q : rx_queues_) {
    if (q->port == intf.port_id_) {
      queues[q->id] = q.get();
    }
  }

  const auto add_rule = [&](uint8_t criteria, DrMatch& mask, DrMatch& value,
                            Destination* destination, uint16_t priority, FlowId tag,
                            const std::string& what) {
    struct mlx5dv_dr_matcher* matcher =
        mlx5dv_dr_matcher_create(dev.table, priority, criteria, dr_param(mask));
    if (matcher == nullptr) {
      DAQIRI_LOG_CRITICAL("Interface '{}': mlx5dv_dr_matcher_create failed for {}: {}", intf.name_,
                          what, strerror(errno));
      return false;
    }
    dev.matchers.push_back(matcher);
    // A tag is delivered with each packet in its completion, so that packets of different flows
    // sharing a queue can be told apart
    struct mlx5dv_dr_action* actions[2];
    size_t num_actions = 0;
    if (tag != 0) {
      struct mlx5dv_dr_action* tag_action = mlx5dv_dr_action_create_tag(tag);
      if (tag_action == nullptr) {
        DAQIRI_LOG_CRITICAL("Interface '{}': mlx5dv_dr_action_create_tag failed for {}: {}",
                            intf.name_, what, strerror(errno));
        return false;
      }
      dev.tag_actions.push_back(tag_action);
      actions[num_actions++] = tag_action;
    }
    actions[num_actions++] = destination->action;
    struct mlx5dv_dr_rule* rule =
        mlx5dv_dr_rule_create(matcher, dr_param(value), num_actions, actions);
    if (rule == nullptr) {
      DAQIRI_LOG_CRITICAL("Interface '{}': mlx5dv_dr_rule_create failed for {}: {}", intf.name_,
                          what, strerror(errno));
      return false;
    }
    dev.rules.push_back(rule);
    return true;
  };

  // Rules match in priority order, a lower value first: the flows in configuration order, then the
  // catch-all
  uint16_t priority = 0;
  for (const auto& flow : intf.rx_.flows_) {
    std::vector<RxQueue*> targets;
    for (const uint16_t id : flow_queues(flow)) {
      const auto it = queues.find(id);
      if (it == queues.end()) {
        DAQIRI_LOG_CRITICAL("RX flow '{}' targets queue {}, which interface '{}' doesn't have",
                            flow.name_, id, intf.name_);
        return false;
      }
      targets.push_back(it->second);
    }
    Destination* destination = rx_destination(dev, targets);
    if (destination == nullptr) {
      return false;
    }

    DrMatch mask;
    DrMatch value;
    auto* mk = reinterpret_cast<uint8_t*>(mask.buf);
    auto* vl = reinterpret_cast<uint8_t*>(value.buf);
    bool any = false;
    const FlowMatch& m = flow.match_;
    if (m.ethernet_match_.match_dst_) {
      const auto [high, low] = split_mac(m.ethernet_match_.dst_);
      DEVX_SET(fte_match_set_lyr_2_4, mk, dmac_47_16, 0xffffffff);
      DEVX_SET(fte_match_set_lyr_2_4, mk, dmac_15_0, 0xffff);
      DEVX_SET(fte_match_set_lyr_2_4, vl, dmac_47_16, high);
      DEVX_SET(fte_match_set_lyr_2_4, vl, dmac_15_0, low);
      any = true;
    }
    if (m.ethernet_match_.match_src_) {
      const auto [high, low] = split_mac(m.ethernet_match_.src_);
      DEVX_SET(fte_match_set_lyr_2_4, mk, smac_47_16, 0xffffffff);
      DEVX_SET(fte_match_set_lyr_2_4, mk, smac_15_0, 0xffff);
      DEVX_SET(fte_match_set_lyr_2_4, vl, smac_47_16, high);
      DEVX_SET(fte_match_set_lyr_2_4, vl, smac_15_0, low);
      any = true;
    }
    // IPv4 and UDP fields imply the IPv4 EtherType and the UDP protocol; so does RSS, which hashes
    // them
    const auto pin_ipv4 = [&]() {
      DEVX_SET(fte_match_set_lyr_2_4, mk, ethertype, 0xffff);
      DEVX_SET(fte_match_set_lyr_2_4, vl, ethertype, MLX5_ETHERTYPE_IPV4);
    };
    const auto pin_ipv4_udp = [&]() {
      pin_ipv4();
      DEVX_SET(fte_match_set_lyr_2_4, mk, ip_protocol, 0xff);
      DEVX_SET(fte_match_set_lyr_2_4, vl, ip_protocol, MLX5_IP_PROTOCOL_UDP);
    };
    if (targets.size() > 1 && m.type_ != FlowMatchType::ETHERNET) {
      pin_ipv4_udp();
    }
    if (m.udp_src_ != 0) {
      pin_ipv4_udp();
      DEVX_SET(fte_match_set_lyr_2_4, mk, udp_sport, 0xffff);
      DEVX_SET(fte_match_set_lyr_2_4, vl, udp_sport, m.udp_src_);
      any = true;
    }
    if (m.udp_dst_ != 0) {
      pin_ipv4_udp();
      DEVX_SET(fte_match_set_lyr_2_4, mk, udp_dport, 0xffff);
      DEVX_SET(fte_match_set_lyr_2_4, vl, udp_dport, m.udp_dst_);
      any = true;
    }
    if (m.ipv4_src_ != INADDR_ANY) {
      pin_ipv4();
      DEVX_SET(fte_match_set_lyr_2_4, mk, src_ipv4_src_ipv6.ipv4_layout.ipv4, 0xffffffff);
      DEVX_SET(fte_match_set_lyr_2_4, vl, src_ipv4_src_ipv6.ipv4_layout.ipv4, ntohl(m.ipv4_src_));
      any = true;
    }
    if (m.ipv4_dst_ != INADDR_ANY) {
      pin_ipv4();
      DEVX_SET(fte_match_set_lyr_2_4, mk, dst_ipv4_dst_ipv6.ipv4_layout.ipv4, 0xffffffff);
      DEVX_SET(fte_match_set_lyr_2_4, vl, dst_ipv4_dst_ipv6.ipv4_layout.ipv4, ntohl(m.ipv4_dst_));
      any = true;
    }
    if (!any) {
      DAQIRI_LOG_CRITICAL("RX flow '{}' on interface '{}' matches no field", flow.name_,
                          intf.name_);
      return false;
    }
    if (flow.id_ > MAX_FLOW_TAG) {
      DAQIRI_LOG_CRITICAL("RX flow '{}' has ID {}; flow IDs are at most {:#x}", flow.name_,
                          flow.id_, MAX_FLOW_TAG);
      return false;
    }
    if (!add_rule(MLX5_DR_MATCH_CRITERIA_OUTER, mask, value, destination, priority++, flow.id_,
                  "flow '" + flow.name_ + "'")) {
      return false;
    }
    DAQIRI_LOG_INFO("RX flow '{}' (ID {}) -> {} queue(s) of interface '{}'", flow.name_, flow.id_,
                    targets.size(), intf.name_);
  }

  if (!intf.rx_.flow_isolation_) {
    RxQueue* first = queues.at(intf.rx_.queues_.front().common_.id_);
    Destination* destination = rx_destination(dev, {first});
    DrMatch mask;
    DrMatch value;
    if (destination == nullptr ||
        !add_rule(0, mask, value, destination, priority, 0, "unmatched packets")) {
      return false;
    }
  }
  DAQIRI_LOG_INFO("Interface '{}': {} RX flow(s) programmed, unmatched packets to {}", intf.name_,
                  intf.rx_.flows_.size(),
                  intf.rx_.flow_isolation_
                      ? "the kernel"
                      : "RX queue '" + intf.rx_.queues_.front().common_.name_ + "'");
  return true;
}

void GpunetioVerbsEngine::destroy_steering(Device& dev) {
  for (auto it = dev.rules.rbegin(); it != dev.rules.rend(); ++it) {
    mlx5dv_dr_rule_destroy(*it);
  }
  dev.rules.clear();
  for (auto it = dev.matchers.rbegin(); it != dev.matchers.rend(); ++it) {
    mlx5dv_dr_matcher_destroy(*it);
  }
  dev.matchers.clear();
  for (auto* action : dev.tag_actions) {
    mlx5dv_dr_action_destroy(action);
  }
  dev.tag_actions.clear();
  for (auto& destination : dev.destinations) {
    if (destination->action != nullptr) {
      mlx5dv_dr_action_destroy(destination->action);
    }
    if (destination->tir != nullptr) {
      mlx5dv_devx_obj_destroy(destination->tir);
    }
    if (destination->rqt != nullptr) {
      mlx5dv_devx_obj_destroy(destination->rqt);
    }
  }
  dev.destinations.clear();
  if (dev.table != nullptr) {
    mlx5dv_dr_table_destroy(dev.table);
    dev.table = nullptr;
  }
  if (dev.domain != nullptr) {
    mlx5dv_dr_domain_destroy(dev.domain);
    dev.domain = nullptr;
  }
  if (dev.td != nullptr) {
    mlx5dv_devx_obj_destroy(dev.td);
    dev.td = nullptr;
  }
}

bool GpunetioVerbsEngine::create_burst_pools() {
  const auto construct = [this](daqiri::ObjectPool* pool) {
    std::vector<void*> objs(pool->size());
    for (auto& obj : objs) {
      if (!pool->get(&obj)) {
        return false;
      }
      burst_objects_.push_back(new (obj) BurstParams());
    }
    for (void* obj : objs) {
      pool->put(obj);
    }
    return true;
  };

  if (!rx_queues_.empty()) {
    rx_meta_pool_ = daqiri::ObjectPool::create(
        "GV_RX_META", std::max<uint32_t>(cfg_.rx_meta_buffers_, 64), sizeof(BurstParams));
    if (rx_meta_pool_ == nullptr || !construct(rx_meta_pool_)) {
      DAQIRI_LOG_CRITICAL("Failed to create the RX burst pool");
      return false;
    }
  }
  if (!tx_queues_.empty()) {
    for (const auto& q : tx_queues_) {
      max_tx_batch_ = std::max(max_tx_batch_, q->batch_size);
    }
    // The packet pointers and lengths of a TX burst follow its BurstParams
    const size_t elt = sizeof(BurstParams) +
                       static_cast<size_t>(max_tx_batch_) * (sizeof(void*) + sizeof(uint32_t));
    tx_meta_pool_ = daqiri::ObjectPool::create("GV_TX_META",
                                               std::max<uint32_t>(cfg_.tx_meta_buffers_, 64), elt);
    if (tx_meta_pool_ == nullptr || !construct(tx_meta_pool_)) {
      DAQIRI_LOG_CRITICAL("Failed to create the TX burst pool");
      return false;
    }
  }
  return true;
}

// Launches a resident kernel on a new stream, one block per queue. The kernel reads the arguments
// of its blocks from pinned host memory.
template <typename Args, typename Launch>
static bool launch_resident_kernel(const char* name, int ordinal, const std::vector<Args>& blocks,
                                   void** args, cudaStream_t* stream, Launch launch) {
  if (blocks.empty()) {
    return true;
  }
  Args* pinned = alloc_pinned<Args>(blocks.size());
  *args = pinned;
  if (pinned == nullptr ||
      cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking) != cudaSuccess) {
    *stream = nullptr;
    DAQIRI_LOG_CRITICAL("GPU {}: could not allocate the {} kernel resources", ordinal, name);
    return false;
  }
  std::copy(blocks.begin(), blocks.end(), pinned);
  const cudaError_t err = launch(pinned, static_cast<uint32_t>(blocks.size()), *stream);
  if (err != cudaSuccess) {
    DAQIRI_LOG_CRITICAL("GPU {}: {} kernel launch for {} queue(s) failed: {}", ordinal, name,
                        blocks.size(), cudaGetErrorString(err));
    return false;
  }
  DAQIRI_LOG_INFO("GPU {}: {} kernel running with {} block(s), one per queue", ordinal, name,
                  blocks.size());
  return true;
}

// One RX kernel with a block per RX queue and one TX kernel with a block per TX queue
bool GpunetioVerbsEngine::start_kernels() {
  if (rx_queues_.empty() && tx_queues_.empty()) {
    return true;
  }
  if (!select_cuda_device(gpu_ordinal_, "starting the gpunetio_verbs kernels")) {
    return false;
  }
  // Under CUDA lazy loading, a kernel launched for the first time waits for the running kernels to
  // exit, and the resident kernels exit only at shutdown: load every kernel of the engine first
  cudaError_t err = gv::load_kernels();
  if (err != cudaSuccess) {
    DAQIRI_LOG_CRITICAL("Could not load the gpunetio_verbs kernels: {}", cudaGetErrorString(err));
    return false;
  }
  const char* loading = std::getenv("CUDA_MODULE_LOADING");
  if (loading == nullptr || std::strcmp(loading, "EAGER") != 0) {
    DAQIRI_LOG_WARN(
        "CUDA lazy loading is on: a kernel the application launches for the first time after "
        "daqiri_init() blocks until shutdown. Set CUDA_MODULE_LOADING=EAGER, or load the "
        "application kernels before daqiri_init() with cudaFuncGetAttributes().");
  }

  std::vector<gv::RxKernelArgs> rx_blocks;
  for (const auto& q : rx_queues_) {
    rx_blocks.push_back(q->kernel_args);
  }
  std::vector<gv::TxKernelArgs> tx_blocks;
  for (const auto& q : tx_queues_) {
    tx_blocks.push_back(q->kernel_args);
  }
  // The RX and TX kernels share the GPU: check that all their blocks fit at once
  err = gv::check_gpu_capacity(static_cast<uint32_t>(rx_blocks.size()),
                               static_cast<uint32_t>(tx_blocks.size()));
  if (err != cudaSuccess) {
    DAQIRI_LOG_CRITICAL(
        "GPU {}: the {} RX and {} TX queues need more blocks than the GPU can run at once: {}",
        gpu_ordinal_, rx_blocks.size(), tx_blocks.size(), cudaGetErrorString(err));
    return false;
  }
  return launch_resident_kernel("RX", gpu_ordinal_, rx_blocks, &rx_args_, &rx_stream_,
                                gv::launch_rx_kernel) &&
         launch_resident_kernel("TX", gpu_ordinal_, tx_blocks, &tx_args_, &tx_stream_,
                                gv::launch_tx_kernel);
}

// Asks the kernels to exit and waits for their streams to drain. The queues of a kernel that
// doesn't stop stay alive: destroying them under a running kernel would fault the GPU.
void GpunetioVerbsEngine::stop_kernels() {
  if (kernels_stuck_) {
    return;
  }
  for (auto& q : rx_queues_) {
    if (q->ctrl != nullptr) {
      store_release(&q->ctrl->exit, 1u);
    }
  }
  for (auto& q : tx_queues_) {
    if (q->ctrl != nullptr) {
      store_release(&q->ctrl->exit, 1u);
    }
  }
  if (rx_stream_ != nullptr && !wait_stream(rx_stream_)) {
    DAQIRI_LOG_CRITICAL("The gpunetio_verbs RX kernel did not stop");
    kernels_stuck_ = true;
  }
  if (tx_stream_ != nullptr && !wait_stream(tx_stream_)) {
    DAQIRI_LOG_CRITICAL("The gpunetio_verbs TX kernel did not stop");
    kernels_stuck_ = true;
  }
}

// One worker thread per cpu_core serves the queues pinned to that core in turn: two busy threads
// on one core would take turns at the scheduler's pace
void GpunetioVerbsEngine::run() {
  std::map<int, std::pair<std::vector<RxQueue*>, std::vector<TxQueue*>>> cores;
  for (auto& q : rx_queues_) {
    cores[q->cpu_core].first.push_back(q.get());
  }
  for (auto& q : tx_queues_) {
    cores[q->cpu_core].second.push_back(q.get());
  }
  workers_running_.store(true);
  for (auto& [core, queues] : cores) {
    workers_.emplace_back(&GpunetioVerbsEngine::worker, this, core, queues.first, queues.second);
  }
}

// ---------------------------------------------------------------------------
// Workers
// ---------------------------------------------------------------------------

void GpunetioVerbsEngine::worker(int core, std::vector<RxQueue*> rx, std::vector<TxQueue*> tx) {
  pin_thread(core, "gpunetio_verbs worker");
  // The flush of GPUDirect RDMA writes applies to the current CUDA context
  if (flush_rx_writes_ && !rx.empty()) {
    (void)select_cuda_device(gpu_ordinal_, "running the RX worker");
  }
  while (workers_running_.load(std::memory_order_relaxed)) {
    bool busy = false;
    for (RxQueue* q : rx) {
      busy = serve_rx_queue(q) || busy;
    }
    for (TxQueue* q : tx) {
      busy = serve_tx_queue(q) || busy;
    }
    if (!busy) {
      cpu_relax();
    }
  }
}

// Turns the bursts published by the RX kernel into BurstParams for the application, and reports
// the packets the application freed back to the kernel, oldest first. Returns whether it did any
// work.
bool GpunetioVerbsEngine::serve_rx_queue(RxQueue* q) {
  const uint64_t ring_mask = q->ring_size - 1;
  bool busy = false;

  uint64_t ready_end = q->consumed;
  while (load_acquire(&q->desc[ready_end & ring_mask].ready) ==
         static_cast<uint32_t>(ready_end + 1)) {
    ready_end++;
  }
  // The NIC wrote the packets of these bursts before their completions: on GPUs that need it,
  // make the packets visible to the application kernels before handing the bursts over
  if (flush_rx_writes_ && ready_end != q->consumed) {
    (void)cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                     CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
  }
  while (q->consumed != ready_end) {
    gv::RxBurstDesc& desc = q->desc[q->consumed & ring_mask];
    BurstParams* burst = nullptr;
    if (!rx_meta_pool_->get(reinterpret_cast<void**>(&burst))) {
      q->pool_empty.fetch_add(1, std::memory_order_relaxed);
      break;
    }
    const uint64_t slot = desc.pkt_seq & ring_mask;
    const uint32_t num = desc.num_pkts;
    uint64_t nbytes = 0;
    for (uint32_t i = 0; i < num; i++) {
      nbytes += q->pkt_len[slot + i];
    }

    auto& hdr = burst->hdr.hdr;
    hdr.num_pkts = num;
    hdr.port_id = static_cast<uint16_t>(q->port);
    hdr.q_id = static_cast<uint16_t>(q->id);
    hdr.num_segs = 1;
    hdr.nbytes = nbytes;
    hdr.first_pkt_addr = reinterpret_cast<uintptr_t>(q->pkt_ptrs[slot]);
    hdr.max_pkt = static_cast<uint32_t>(q->ring_size);
    hdr.max_pkt_size = q->stride;
    hdr.gpu_pkt0_idx = static_cast<uint32_t>(slot);
    hdr.gpu_pkt0_addr = hdr.first_pkt_addr;
    hdr.burst_flags = GV_RX_BURST_FLAG;
    // The bursts never wrap around the ring, so their packets index straight into it
    burst->pkts[0] = &q->pkt_ptrs[slot];
    burst->pkt_lens[0] = &q->pkt_len[slot];
    for (size_t s = 1; s < burst->pkts.size(); s++) {
      burst->pkts[s] = nullptr;
      burst->pkt_lens[s] = nullptr;
    }
    burst->pkt_extra_info = nullptr;
    burst->event = nullptr;
    set_priv(burst, q->consumed, q);

    const uint64_t desc_slot = q->consumed & ring_mask;
    q->desc_pkts[desc_slot] = num;
    q->desc_freed[desc_slot].store(0, std::memory_order_relaxed);
    if (!q->ring->enqueue(burst)) {
      rx_meta_pool_->put(burst);
      q->ring_full.fetch_add(1, std::memory_order_relaxed);
      break;
    }
    q->consumed++;
    q->bursts.fetch_add(1, std::memory_order_relaxed);
    q->pkts.fetch_add(num, std::memory_order_relaxed);
    q->bytes.fetch_add(nbytes, std::memory_order_relaxed);
    busy = true;
  }

  const uint64_t released_before = q->released_pkts;
  while (q->release_cursor < q->consumed) {
    const uint64_t desc_slot = q->release_cursor & ring_mask;
    if (q->desc_freed[desc_slot].load(std::memory_order_acquire) < q->desc_pkts[desc_slot]) {
      break;
    }
    q->released_pkts += q->desc_pkts[desc_slot];
    q->release_cursor++;
  }
  if (q->released_pkts != released_before) {
    store_release(&q->ctrl->released_pkts, q->released_pkts);
    busy = true;
  }

  if (!busy && !q->error_logged) {
    const uint32_t error = load_acquire(&q->ctrl->error);
    if (error != gv::KERNEL_OK) {
      q->error_logged = true;
      DAQIRI_LOG_ERROR("RX queue '{}' stopped: {} (syndrome {:#x})", q->name,
                       kernel_error_str(error), q->ctrl->syndrome);
    }
  }
  return busy;
}

// Hands the bursts sent by the application to the TX kernel, then retires them once the NIC
// completes them: their TX slots and BurstParams become free again. Returns whether it did any
// work.
bool GpunetioVerbsEngine::serve_tx_queue(TxQueue* q) {
  bool busy = false;

  while (!q->failed.load(std::memory_order_relaxed) && q->inflight.size() < TX_DESC_RING) {
    void* obj = nullptr;
    if (!q->send_ring->dequeue(&obj)) {
      break;
    }
    auto* burst = static_cast<BurstParams*>(obj);
    BurstPriv priv{};
    (void)get_priv(burst, &priv);
    const uint32_t num = static_cast<uint32_t>(burst->hdr.hdr.num_pkts);
    if (num == 0) {
      // Nothing to send, and no completion would retire it
      q->retire(priv.seq, priv.reserved);
      tx_meta_pool_->put(burst);
      continue;
    }
    uint64_t nbytes = 0;
    for (uint32_t i = 0; i < num; i++) {
      gv::TxPacket& pkt = q->pkts[(priv.seq + i) % q->num_slots];
      pkt.addr = reinterpret_cast<uint64_t>(burst->pkts[0][i]);
      pkt.len = burst->pkt_lens[0][i];
      nbytes += pkt.len;
    }

    gv::TxBurstDesc& desc = q->desc[q->published & (TX_DESC_RING - 1)];
    desc.slot_start = priv.seq;
    desc.num_pkts = num;
    store_release(&desc.ready, static_cast<uint32_t>(q->published + 1));
    q->published++;
    q->inflight.push_back(burst);
    q->sent_bursts.fetch_add(1, std::memory_order_relaxed);
    q->sent_pkts.fetch_add(num, std::memory_order_relaxed);
    q->sent_bytes.fetch_add(nbytes, std::memory_order_relaxed);
    busy = true;
  }

  const uint64_t done = load_acquire(&q->ctrl->completed_bursts);
  while (q->reaped < done && !q->inflight.empty()) {
    BurstParams* burst = q->inflight.front();
    q->inflight.pop_front();
    BurstPriv done_priv{};
    if (get_priv(burst, &done_priv)) {
      q->retire(done_priv.seq, done_priv.reserved);
    }
    tx_meta_pool_->put(burst);
    q->reaped++;
    busy = true;
  }

  if (!busy && !q->failed.load(std::memory_order_relaxed)) {
    const uint32_t error = load_acquire(&q->ctrl->error);
    if (error != gv::KERNEL_OK && !q->failed.exchange(true)) {
      DAQIRI_LOG_ERROR("TX queue '{}' stopped: {} (syndrome {:#x})", q->name,
                       kernel_error_str(error), q->ctrl->syndrome);
    }
  }
  return busy;
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------
void GpunetioVerbsEngine::shutdown() {
  workers_running_.store(false);
  for (auto& worker : workers_) {
    worker.join();
  }
  workers_.clear();
  stop_kernels();

  // The steering rules reference the receive queues: remove them first
  for (auto& dev : devices_) {
    destroy_steering(*dev);
  }

  // A stuck kernel may still use the queues and the memory regions: leave them in place
  for (auto& q : rx_queues_) {
    if (!kernels_stuck_) {
      if (q->rq != nullptr) {
        doca_verbs_eth_rq_destroy(q->rq);
        q->rq = nullptr;
      }
      destroy_rings(gpu_, q->rings);
      free_pinned(q->ctrl);
      free_pinned(q->desc);
      free_pinned(q->pkt_len);
      free_pinned(q->flow_tag);
    }
    daqiri::Ring::free(q->ring);
  }
  for (auto& q : tx_queues_) {
    if (!kernels_stuck_) {
      if (q->sq != nullptr) {
        doca_verbs_eth_sq_destroy(q->sq);
        q->sq = nullptr;
      }
      destroy_rings(gpu_, q->rings);
      free_pinned(q->pkts);
      free_pinned(q->desc);
      free_pinned(q->ctrl);
    }
    daqiri::Ring::free(q->send_ring);
  }
  rx_queues_.clear();
  tx_queues_.clear();

  if (!kernels_stuck_) {
    for (auto* stream : {&rx_stream_, &tx_stream_}) {
      if (*stream != nullptr) {
        cudaStreamDestroy(*stream);
        *stream = nullptr;
      }
    }
    free_pinned(rx_args_);
    free_pinned(tx_args_);
    rx_args_ = nullptr;
    tx_args_ = nullptr;
  }

  for (auto& dev : devices_) {
    if (!kernels_stuck_) {
      for (auto& [name, mr] : dev->mrs) {
        (void)name;
        ibv_dereg_mr(mr);
      }
      dev->mrs.clear();
      // The imported DOCA Verbs PD doesn't own the ibv_pd
      if (dev->dev != nullptr) {
        doca_dev_close(dev->dev);
      }
      if (dev->vpd != nullptr) {
        doca_verbs_pd_destroy(dev->vpd);
      }
      if (dev->pd != nullptr) {
        ibv_dealloc_pd(dev->pd);
      }
      if (dev->vctx != nullptr) {
        doca_verbs_context_destroy(dev->vctx);
      }
      if (dev->ctx != nullptr) {
        ibv_close_device(dev->ctx);
      }
    }
  }
  devices_.clear();
  port_devices_.clear();

  if (!kernels_stuck_) {
    if (gpu_ != nullptr) {
      doca_gpu_destroy(gpu_);
      gpu_ = nullptr;
    }
    free_memory_regions();
  } else {
    // Leak the memory regions on purpose: the base destructor would free them under the kernel,
    // and freeing GPU memory also waits for it
    ar_.clear();
  }

  for (BurstParams* burst : burst_objects_) {
    burst->~BurstParams();
  }
  burst_objects_.clear();
  daqiri::ObjectPool::free(rx_meta_pool_);
  daqiri::ObjectPool::free(tx_meta_pool_);
  rx_meta_pool_ = nullptr;
  tx_meta_pool_ = nullptr;
  max_tx_batch_ = 0;
  flush_rx_writes_ = false;
  initialized_ = false;
}

void GpunetioVerbsEngine::print_stats() {
  for (const auto& q : rx_queues_) {
    const uint64_t gpu_pkts = q->ctrl != nullptr ? load_acquire(&q->ctrl->pkts) : 0;
    DAQIRI_LOG_INFO(
        "gpunetio_verbs RX port {} q{} '{}': bursts={} pkts={} bytes={} gpu_pkts={} released={} "
        "burst_pool_empty={} app_ring_full={}",
        q->port, q->id, q->name, q->bursts.load(), q->pkts.load(), q->bytes.load(), gpu_pkts,
        q->released_pkts, q->pool_empty.load(), q->ring_full.load());
  }
  for (const auto& q : tx_queues_) {
    const uint64_t done = q->ctrl != nullptr ? load_acquire(&q->ctrl->completed_bursts) : 0;
    DAQIRI_LOG_INFO(
        "gpunetio_verbs TX port {} q{} '{}': bursts={} pkts={} bytes={} completed_bursts={} "
        "send_ring_full={}",
        q->port, q->id, q->name, q->sent_bursts.load(), q->sent_pkts.load(), q->sent_bytes.load(),
        done, q->send_ring_full.load());
  }
}

Status GpunetioVerbsEngine::get_mac_addr(int port, char* mac) {
  if (mac == nullptr) {
    return Status::NULL_PTR;
  }
  if (port < 0 || static_cast<size_t>(port) >= port_devices_.size() ||
      port_devices_[port] == nullptr) {
    DAQIRI_LOG_ERROR("get_mac_addr: invalid port {}", port);
    return Status::INVALID_PARAMETER;
  }
  const doca_error_t ret =
      doca_devinfo_get_mac_addr(doca_dev_as_devinfo(port_devices_[port]->dev),
                                reinterpret_cast<uint8_t*>(mac), DOCA_DEVINFO_MAC_ADDR_SIZE);
  if (ret != DOCA_SUCCESS) {
    DAQIRI_LOG_ERROR("get_mac_addr: failed to read the MAC address of port {}: {}", port,
                     doca_err(ret));
    return Status::GENERIC_FAILURE;
  }
  return Status::SUCCESS;
}

GpunetioVerbsEngine::RxQueue* GpunetioVerbsEngine::find_rx_queue(int port, int q) const {
  for (const auto& rxq : rx_queues_) {
    if (rxq->port == port && rxq->id == q) {
      return rxq.get();
    }
  }
  return nullptr;
}

GpunetioVerbsEngine::TxQueue* GpunetioVerbsEngine::find_tx_queue(int port, int q) const {
  for (const auto& txq : tx_queues_) {
    if (txq->port == port && txq->id == q) {
      return txq.get();
    }
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Packet accessors
// ---------------------------------------------------------------------------
void* GpunetioVerbsEngine::get_packet_ptr(BurstParams* burst, int idx) {
  return burst->pkts[0][idx];
}

uint32_t GpunetioVerbsEngine::get_packet_length(BurstParams* burst, int idx) {
  return burst->pkt_lens[0][idx];
}

void* GpunetioVerbsEngine::get_segment_packet_ptr(BurstParams* burst, int seg, int idx) {
  return burst->pkts[seg][idx];
}

uint32_t GpunetioVerbsEngine::get_segment_packet_length(BurstParams* burst, int seg, int idx) {
  return burst->pkt_lens[seg][idx];
}

FlowId GpunetioVerbsEngine::get_packet_flow_id(BurstParams* burst, int idx) {
  BurstPriv priv{};
  if (burst == nullptr || !(burst->hdr.hdr.burst_flags & GV_RX_BURST_FLAG) ||
      !get_priv(burst, &priv)) {
    return 0;
  }
  const auto* q = static_cast<const RxQueue*>(priv.queue);
  return q->flow_tag[burst->hdr.hdr.gpu_pkt0_idx + idx];
}

Status GpunetioVerbsEngine::get_packet_rx_timestamp(BurstParams* burst, int idx,
                                                    uint64_t* timestamp_ns) {
  (void)burst;
  (void)idx;
  (void)timestamp_ns;
  return Status::NOT_SUPPORTED;
}

void* GpunetioVerbsEngine::get_packet_extra_info(BurstParams* burst, int idx) {
  (void)burst;
  (void)idx;
  return nullptr;
}

uint64_t GpunetioVerbsEngine::get_burst_tot_byte(BurstParams* burst) {
  uint64_t total = 0;
  for (size_t i = 0; i < burst->hdr.hdr.num_pkts; i++) {
    total += burst->pkt_lens[0][i];
  }
  return total;
}

// ---------------------------------------------------------------------------
// RX
// ---------------------------------------------------------------------------
Status GpunetioVerbsEngine::get_rx_burst(BurstParams** burst, int port, int q) {
  if (burst == nullptr) {
    return Status::NULL_PTR;
  }
  *burst = nullptr;
  RxQueue* rxq = find_rx_queue(port, q);
  if (rxq == nullptr) {
    return Status::INVALID_PARAMETER;
  }
  void* obj = nullptr;
  if (!rxq->ring->dequeue(&obj)) {
    return Status::NOT_READY;
  }
  *burst = static_cast<BurstParams*>(obj);
  return Status::SUCCESS;
}

void GpunetioVerbsEngine::free_all_packets(BurstParams* burst) {
  if (burst == nullptr) {
    return;
  }
  BurstPriv priv{};
  if (!get_priv(burst, &priv)) {
    return;
  }
  const auto& hdr = burst->hdr.hdr;

  if (hdr.burst_flags & GV_TX_BURST_FLAG) {
    // A TX burst the application allocated but did not send: its slots are retired, and become
    // free once the earlier ones are
    if (priv.reserved != 0) {
      static_cast<TxQueue*>(priv.queue)->retire(priv.seq, priv.reserved);
      set_priv(burst, priv.seq, priv.queue, 0);
    }
    return;
  }
  if (hdr.burst_flags & GV_RX_BURST_FLAG) {
    auto* q = static_cast<RxQueue*>(priv.queue);
    q->desc_freed[priv.seq & (q->ring_size - 1)].store(static_cast<uint32_t>(hdr.num_pkts),
                                                       std::memory_order_release);
  }
}

void GpunetioVerbsEngine::free_packet(BurstParams* burst, int pkt) {
  (void)pkt;
  if (burst == nullptr || !(burst->hdr.hdr.burst_flags & GV_RX_BURST_FLAG)) {
    return;
  }
  BurstPriv priv{};
  if (!get_priv(burst, &priv)) {
    return;
  }
  auto* q = static_cast<RxQueue*>(priv.queue);
  q->desc_freed[priv.seq & (q->ring_size - 1)].fetch_add(1, std::memory_order_release);
}

void GpunetioVerbsEngine::free_all_segment_packets(BurstParams* burst, int seg) {
  if (seg == 0) {
    free_all_packets(burst);
  }
}

void GpunetioVerbsEngine::free_packet_segment(BurstParams* burst, int seg, int pkt) {
  if (seg == 0) {
    free_packet(burst, pkt);
  }
}

void GpunetioVerbsEngine::free_rx_burst(BurstParams* burst) {
  if (burst != nullptr && rx_meta_pool_ != nullptr) {
    rx_meta_pool_->put(burst);
  }
}

void GpunetioVerbsEngine::free_rx_metadata(BurstParams* burst) {
  free_rx_burst(burst);
}

// ---------------------------------------------------------------------------
// TX
// ---------------------------------------------------------------------------
BurstParams* GpunetioVerbsEngine::create_tx_burst_params() {
  BurstParams* burst = nullptr;
  if (tx_meta_pool_ == nullptr || !tx_meta_pool_->get(reinterpret_cast<void**>(&burst))) {
    return nullptr;
  }
  auto* arrays = reinterpret_cast<uint8_t*>(burst) + sizeof(BurstParams);
  burst->hdr.hdr.num_pkts = 0;
  burst->hdr.hdr.num_segs = 1;
  burst->hdr.hdr.burst_flags = GV_TX_BURST_FLAG;
  burst->pkts[0] = reinterpret_cast<void**>(arrays);
  burst->pkt_lens[0] = reinterpret_cast<uint32_t*>(arrays + max_tx_batch_ * sizeof(void*));
  for (size_t s = 1; s < burst->pkts.size(); s++) {
    burst->pkts[s] = nullptr;
    burst->pkt_lens[s] = nullptr;
  }
  std::memset(burst->hdr.custom_burst_data, 0, sizeof(BurstPriv));
  return burst;
}

bool GpunetioVerbsEngine::is_tx_burst_available(BurstParams* burst) {
  if (burst == nullptr) {
    return false;
  }
  TxQueue* q = find_tx_queue(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
  if (q == nullptr || burst->hdr.hdr.num_pkts > max_tx_batch_) {
    return false;
  }
  const uint64_t in_flight = q->alloc_head - q->completed_tail.load(std::memory_order_acquire);
  return q->num_slots - in_flight >= burst->hdr.hdr.num_pkts;
}

Status GpunetioVerbsEngine::get_tx_packet_burst(BurstParams* burst) {
  if (burst == nullptr) {
    return Status::NULL_PTR;
  }
  TxQueue* q = find_tx_queue(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
  if (q == nullptr) {
    return Status::INVALID_PARAMETER;
  }
  const uint64_t num = burst->hdr.hdr.num_pkts;
  if (num > max_tx_batch_) {
    DAQIRI_LOG_ERROR("TX queue '{}': burst of {} packets exceeds the batch size {}", q->name, num,
                     max_tx_batch_);
    return Status::INVALID_PARAMETER;
  }
  const uint64_t in_flight = q->alloc_head - q->completed_tail.load(std::memory_order_acquire);
  if (q->num_slots - in_flight < num) {
    return Status::NO_FREE_PACKET_BUFFERS;
  }

  for (uint64_t i = 0; i < num; i++) {
    burst->pkts[0][i] = q->slot_base + ((q->alloc_head + i) % q->num_slots) * q->slot_size;
    burst->pkt_lens[0][i] = 0;
  }
  burst->hdr.hdr.num_segs = 1;
  burst->hdr.hdr.burst_flags = GV_TX_BURST_FLAG;
  set_priv(burst, q->alloc_head, q, static_cast<uint32_t>(num));
  q->alloc_head += num;
  return Status::SUCCESS;
}

Status GpunetioVerbsEngine::set_packet_lengths(BurstParams* burst, int idx,
                                               const std::initializer_list<int>& lens) {
  if (lens.size() != 1) {
    return Status::INVALID_PARAMETER;
  }
  burst->pkt_lens[0][idx] = static_cast<uint32_t>(*lens.begin());
  return Status::SUCCESS;
}

Status GpunetioVerbsEngine::set_packet_tx_time(BurstParams* burst, int idx, uint64_t time) {
  (void)burst;
  (void)idx;
  (void)time;
  return Status::NOT_SUPPORTED;
}

// The TX slots are usually in GPU memory: write through CUDA
Status GpunetioVerbsEngine::write_packet(BurstParams* burst, int idx, size_t offset,
                                         const void* data, size_t len) {
  auto* dst = static_cast<uint8_t*>(burst->pkts[0][idx]) + offset;
  return cudaMemcpy(dst, data, len, cudaMemcpyDefault) == cudaSuccess ? Status::SUCCESS
                                                                      : Status::GENERIC_FAILURE;
}

Status GpunetioVerbsEngine::set_eth_header(BurstParams* burst, int idx, char* dst_addr) {
  struct ethhdr eth {};
  std::memcpy(eth.h_dest, dst_addr, ETH_ALEN);
  if (get_mac_addr(burst->hdr.hdr.port_id, reinterpret_cast<char*>(eth.h_source)) !=
      Status::SUCCESS) {
    return Status::GENERIC_FAILURE;
  }
  eth.h_proto = htons(ETH_P_IP);
  return write_packet(burst, idx, 0, &eth, sizeof(eth));
}

Status GpunetioVerbsEngine::set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                                            unsigned int src_host, unsigned int dst_host) {
  struct iphdr ip {};
  ip.version = 4;
  ip.ihl = 5;
  ip.tot_len = htons(static_cast<uint16_t>(ip_len));
  ip.ttl = 64;
  ip.protocol = proto;
  ip.saddr = src_host;
  ip.daddr = dst_host;
  return write_packet(burst, idx, sizeof(struct ethhdr), &ip, sizeof(ip));
}

Status GpunetioVerbsEngine::set_udp_header(BurstParams* burst, int idx, int udp_len,
                                           uint16_t src_port, uint16_t dst_port) {
  struct udphdr udp {};
  udp.source = htons(src_port);
  udp.dest = htons(dst_port);
  udp.len = htons(static_cast<uint16_t>(udp_len));
  return write_packet(burst, idx, sizeof(struct ethhdr) + sizeof(struct iphdr), &udp, sizeof(udp));
}

Status GpunetioVerbsEngine::set_udp_payload(BurstParams* burst, int idx, void* data, int len) {
  return write_packet(burst, idx, sizeof(UDPIPV4Pkt), data, static_cast<size_t>(len));
}

Status GpunetioVerbsEngine::send_tx_burst(BurstParams* burst) {
  if (burst == nullptr) {
    return Status::NULL_PTR;
  }
  TxQueue* q = find_tx_queue(burst->hdr.hdr.port_id, burst->hdr.hdr.q_id);
  if (q == nullptr) {
    return Status::INVALID_PARAMETER;
  }
  if (q->failed.load(std::memory_order_relaxed)) {
    return Status::GENERIC_FAILURE;
  }
  // The NIC reads a byte count of 0 as 2 GB, and a packet longer than its slot would send the
  // next one: either stops the queue with a completion error
  for (size_t i = 0; i < burst->hdr.hdr.num_pkts; i++) {
    const uint32_t len = burst->pkt_lens[0][i];
    if (len == 0 || len > q->slot_size) {
      DAQIRI_LOG_ERROR(
          "TX queue '{}': packet {} of the burst has length {}; valid lengths are 1 to {}", q->name,
          i, len, q->slot_size);
      return Status::INVALID_PARAMETER;
    }
  }
  if (!q->send_ring->enqueue(burst)) {
    // The worker is behind: give back the slots of this burst
    BurstPriv priv{};
    if (get_priv(burst, &priv)) {
      q->retire(priv.seq, priv.reserved);
    }
    q->send_ring_full.fetch_add(1, std::memory_order_relaxed);
    tx_meta_pool_->put(burst);
    return Status::NO_SPACE_AVAILABLE;
  }
  return Status::SUCCESS;
}

void GpunetioVerbsEngine::free_tx_burst(BurstParams* burst) {
  if (burst != nullptr && tx_meta_pool_ != nullptr) {
    tx_meta_pool_->put(burst);
  }
}

void GpunetioVerbsEngine::free_tx_metadata(BurstParams* burst) {
  free_tx_burst(burst);
}

}  // namespace daqiri
