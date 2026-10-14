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

#include "src/engines/gpunetio_rdma/daqiri_gpunetio_rdma_engine.h"

#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_verbs_def.h>
#include <doca_rdma_bridge.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#include <arpa/inet.h>
#include <cuda.h>
#include <endian.h>
#include <fcntl.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <rdma/rdma_cma.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <set>

#include "src/engines/gpunetio_rdma/daqiri_gpunetio_rdma_kernels.h"

namespace daqiri {

namespace gr = gpunetio_rdma;

namespace {

// Work requests of each send and receive queue, a power of two. Every work request asks for a
// completion, so the completion queues have the same size.
constexpr uint32_t QP_DEPTH = 1024;
constexpr uint32_t QP_DEPTH_MASK = QP_DEPTH - 1;
constexpr size_t SQ_WQE_SIZE = 64;
constexpr size_t RQ_WQE_SIZE = 16;
constexpr size_t CQE_SIZE = 64;
constexpr size_t CQ_DBR_SIZE = 8;
// Largest message of one work request with DOCA GPUNetIO
constexpr uint64_t MAX_MESSAGE_SIZE = 1ULL << 30;
// Messages per burst
constexpr uint32_t MAX_BURST = QP_DEPTH;
// Bursts between the application and a slot worker, in each direction
constexpr uint32_t APP_RING = 2048;
// Burst metadata objects and their packet arrays
constexpr uint32_t BURST_POOL = (1U << 11) - 1U;
constexpr auto DRAIN_TIMEOUT = std::chrono::seconds(5);
constexpr auto CLOSE_TIMEOUT = std::chrono::seconds(10);
constexpr int CM_TIMEOUT_MS = 2000;
// Responder resources and initiator depth offered to the peer, as the ibverbs RDMA engine does
constexpr uint8_t RDMA_READ_DEPTH = 1;

enum class SlotHostState : uint32_t {
  FREE = 0,        // no connection, queue pair in reset
  CONNECTING = 1,  // queue pair being connected
  ACTIVE = 2,      // connected, the worker and the kernel serve it
  CLOSING = 3,     // the worker drains it
};

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

size_t page_size() {
  static const size_t size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return size;
}

size_t round_up(size_t value, size_t align) {
  return (value + align - 1) / align * align;
}

const char* doca_err(doca_error_t err) {
  return doca_error_get_descr(err);
}

int env_min_rnr_timer() {
  // Same default and override as the ibverbs RDMA engine: rdma_cm leaves min_rnr_timer at 0, which
  // means 655 ms, so a short receive underrun on the responder stalls the sender for that long
  int rnr_timer = 12;  // 0.64 ms
  if (const char* env = std::getenv("DAQIRI_RDMA_MIN_RNR_TIMER")) {
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end != env && *end == '\0' && parsed >= 0 && parsed <= 31) {
      rnr_timer = static_cast<int>(parsed);
    } else {
      DAQIRI_LOG_WARN(
          "Ignoring invalid DAQIRI_RDMA_MIN_RNR_TIMER='{}' (expected integer 0-31); using {}", env,
          rnr_timer);
    }
  }
  return rnr_timer;
}

void reset_burst_metadata(BurstParams* burst) {
  burst->transport_hdr = BurstTransportHeader{};
  burst->pkts.fill(nullptr);
  burst->pkt_lens.fill(nullptr);
  burst->pkt_extra_info = nullptr;
  burst->event = nullptr;
}

const char* syndrome_str(uint32_t syndrome) {
  switch (syndrome) {
    case MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR:
      return "local length error (message bigger than the receive buffer?)";
    case MLX5_CQE_SYNDROME_LOCAL_PROT_ERR:
      return "local protection error";
    case MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR:
      return "remote invalid request";
    case MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR:
      return "remote access error";
    case MLX5_CQE_SYNDROME_REMOTE_OP_ERR:
      return "remote operation error";
    case MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR:
      return "transport retry counter exceeded";
    case MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR:
      return "RNR retry counter exceeded (no receive posted by the peer)";
    default:
      return "completion error";
  }
}

}  // namespace

// One RDMA device (ibv_context) used by at least one interface
struct GpunetioRdmaEngine::Device {
  std::string name;
  struct ibv_context* ctx = nullptr;  // owned by librdmacm
  struct ibv_pd* pd = nullptr;
  struct doca_verbs_context* vctx = nullptr;
  struct doca_verbs_pd* vpd = nullptr;
  struct doca_dev* dev = nullptr;
  std::unordered_map<std::string, struct ibv_mr*> mrs;
  // Target of the memory consistency WQEs of pre-Hopper GPUs
  void* dump_buf = nullptr;
  struct ibv_mr* dump_mr = nullptr;
};

// One RDMA server listening on an interface
struct GpunetioRdmaEngine::Listener {
  int if_idx = 0;
  std::string key;  // "address:port"
  struct rdma_cm_id* id = nullptr;
};

// One queue slot: a queue pair, its kernel block and its worker thread, serving one connection at a
// time. Server interfaces have one slot per RX queue, client interfaces one per TX queue.
struct GpunetioRdmaEngine::Slot {
  int index = 0;
  int if_idx = 0;
  int queue_idx = 0;
  bool server = false;
  int cpu_core = -1;
  std::string name;
  Device* dev = nullptr;

  // DOCA Verbs RC queue pair with its rings and completion queues in GPU memory
  struct doca_verbs_cq* cq_sq = nullptr;
  struct doca_verbs_cq* cq_rq = nullptr;
  struct doca_verbs_qp* qp = nullptr;
  struct doca_uar* uar = nullptr;
  struct doca_umem* qp_umem = nullptr;
  struct doca_umem* dbr_umem = nullptr;
  struct doca_umem* cq_sq_umem = nullptr;
  struct doca_umem* cq_rq_umem = nullptr;
  void* qp_ring = nullptr;
  void* dbr = nullptr;
  void* cq_sq_ring = nullptr;
  void* cq_rq_ring = nullptr;
  struct doca_gpu_verbs_qp* gpu_qp = nullptr;
  struct doca_gpu_dev_verbs_qp* dev_qp = nullptr;
  uint32_t qpn = 0;

  // Shared with the kernel, in pinned host memory
  gr::SlotCtrl* ctrl = nullptr;
  gr::BurstDesc* sq_desc = nullptr;
  gr::BurstDesc* rq_desc = nullptr;
  gr::Packet* sq_pkts = nullptr;
  gr::Packet* rq_pkts = nullptr;
  uint32_t sq_pkt_limit = QP_DEPTH;  // send messages in flight, leaving room for dump WQEs

  // Connection, changed under slots_mutex_
  std::atomic<uint32_t> host_state{static_cast<uint32_t>(SlotHostState::FREE)};
  struct rdma_cm_id* cm_id = nullptr;
  std::atomic<uintptr_t> conn_id{0};
  uint32_t generation = 0;
  bool claimed = false;  // server: handed to the application by rdma_get_server_conn_id()
  std::atomic<bool> close_requested{false};
  // Application calls using the rings; the worker closes the slot only once they are done
  std::atomic<uint32_t> app_users{0};

  // Bursts sent by the application, and completed bursts returned to it
  daqiri::Ring* app_tx = nullptr;
  daqiri::Ring* app_rx = nullptr;

  // Worker state
  struct Inflight {
    BurstParams* burst;
    uint32_t first_pkt;
    uint32_t num_pkts;
  };
  std::deque<Inflight> sq_inflight;
  std::deque<Inflight> rq_inflight;
  std::deque<BurstParams*> deferred;     // waiting for room on the GPU rings
  std::deque<BurstParams*> completions;  // waiting for room in app_rx
  uint64_t sq_published = 0;
  uint64_t rq_published = 0;
  uint64_t sq_reaped = 0;
  uint64_t rq_reaped = 0;
  uint32_t sq_pkt_head = 0;
  uint32_t rq_pkt_head = 0;
  uint32_t sq_pkts_inflight = 0;
  uint32_t rq_pkts_inflight = 0;
  bool closing = false;  // completions are freed instead of returned
  size_t closing_freed = 0;
  uint32_t logged_error = 0;
  std::thread thread;

  std::atomic<uint64_t> connections{0};
  std::atomic<uint64_t> send_msgs{0};
  std::atomic<uint64_t> send_bytes{0};
  std::atomic<uint64_t> recv_msgs{0};
  std::atomic<uint64_t> recv_bytes{0};
  std::atomic<uint64_t> failed_bursts{0};
};

GpunetioRdmaEngine::GpunetioRdmaEngine() = default;

GpunetioRdmaEngine::~GpunetioRdmaEngine() {
  shutdown();
}

bool GpunetioRdmaEngine::set_config_and_initialize(const NetworkConfig& cfg) {
  if (initialized_) {
    DAQIRI_LOG_ERROR("gpunetio_rdma engine is already initialized; call shutdown() first");
    return false;
  }
  cfg_ = cfg;
  if (!validate_config()) {
    DAQIRI_LOG_CRITICAL("Config validation failed");
    return false;
  }
  initialize();
  return initialized_;
}

bool GpunetioRdmaEngine::validate_config() const {
  return Engine::validate_config();
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void GpunetioRdmaEngine::initialize() {
  DAQIRI_LOG_INFO("Initializing gpunetio_rdma engine (RoCE, experimental)");

  int if_num = 0;
  for (auto& intf : cfg_.ifs_) {
    intf.port_id_ = static_cast<uint16_t>(if_num++);
  }
  // Round buffers up to the GPU page size, as the ibverbs RDMA engine does
  for (auto& mr : cfg_.mrs_) {
    const size_t align = std::max<size_t>(get_alignment(mr.second.kind_), GPU_PAGE_SIZE);
    mr.second.adj_size_ = round_up(mr.second.buf_size_, align);
  }

  if (!setup_gpu() || !setup_devices() || allocate_memory_regions() != Status::SUCCESS ||
      !register_memory_regions() || !setup_pools() || !create_slots() || !start_kernel() ||
      !start_listeners()) {
    DAQIRI_LOG_CRITICAL("gpunetio_rdma engine initialization failed");
    destroy_resources();
    return;
  }

  workers_running_.store(true);
  for (auto& slot : slots_) {
    slot->thread = std::thread(&GpunetioRdmaEngine::worker, this, slot.get());
  }
  cm_running_.store(true);
  cm_thread_ = std::thread(&GpunetioRdmaEngine::run, this);

  initialized_ = true;
  DAQIRI_LOG_INFO("gpunetio_rdma engine initialized: {} queue slot(s) on CUDA device {}",
                  slots_.size(), gpu_ordinal_);
}

// The kernel and the queue rings of every slot live on one GPU: the one of the device memory
// regions, or CUDA device 0
bool GpunetioRdmaEngine::setup_gpu() {
  gpu_ordinal_ = 0;
  for (const auto& [name, mr] : cfg_.mrs_) {
    (void)name;
    if (mr.kind_ == MemoryKind::DEVICE) {
      gpu_ordinal_ = mr.affinity_;
      break;
    }
  }
  if (!select_cuda_device(gpu_ordinal_, "initializing the gpunetio_rdma engine")) {
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
  const doca_error_t ret = doca_gpu_create(bus_id, &gpu_);
  if (ret != DOCA_SUCCESS) {
    gpu_ = nullptr;
    DAQIRI_LOG_CRITICAL("doca_gpu_create({}) failed: {}", bus_id, doca_err(ret));
    return false;
  }
  DAQIRI_LOG_INFO("CUDA device {} ({}) runs the gpunetio_rdma datapath", gpu_ordinal_, bus_id);
  return true;
}

// Finds the RDMA device of every interface address and opens it with DOCA Verbs
bool GpunetioRdmaEngine::setup_devices() {
  int num_devices = 0;
  // Keeps every device context open until shutdown
  cm_devices_ = rdma_get_devices(&num_devices);
  if (cm_devices_ == nullptr || num_devices == 0) {
    DAQIRI_LOG_CRITICAL("No RDMA-capable devices found");
    return false;
  }
  cm_channel_ = rdma_create_event_channel();
  if (cm_channel_ == nullptr) {
    DAQIRI_LOG_CRITICAL("Could not create an RDMA CM event channel: {}", strerror(errno));
    return false;
  }
  const int flags = fcntl(cm_channel_->fd, F_GETFL);
  fcntl(cm_channel_->fd, F_SETFL, flags | O_NONBLOCK);

  if_device_.assign(cfg_.ifs_.size(), nullptr);
  for (size_t i = 0; i < cfg_.ifs_.size(); i++) {
    const auto& intf = cfg_.ifs_[i];
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    if (inet_pton(AF_INET, intf.address_.c_str(), &addr.sin_addr) != 1) {
      DAQIRI_LOG_CRITICAL("Interface '{}': '{}' is not an IPv4 address", intf.name_, intf.address_);
      return false;
    }
    // Binding a CM ID to the address resolves its RDMA device
    struct rdma_cm_id* probe = nullptr;
    if (rdma_create_id(cm_channel_, &probe, nullptr, RDMA_PS_TCP) != 0) {
      DAQIRI_LOG_CRITICAL("rdma_create_id failed: {}", strerror(errno));
      return false;
    }
    const int bound = rdma_bind_addr(probe, reinterpret_cast<struct sockaddr*>(&addr));
    struct ibv_context* ctx = bound == 0 ? probe->verbs : nullptr;
    rdma_destroy_id(probe);
    if (ctx == nullptr) {
      DAQIRI_LOG_CRITICAL("Interface '{}': address {} is not on an RDMA device", intf.name_,
                          intf.address_);
      return false;
    }

    Device* device = nullptr;
    for (auto& d : devices_) {
      if (d->ctx == ctx) {
        device = d.get();
      }
    }
    if (device == nullptr) {
      auto d = std::make_unique<Device>();
      d->ctx = ctx;
      d->name = ibv_get_device_name(ctx->device);
      d->pd = ibv_alloc_pd(ctx);
      if (d->pd == nullptr) {
        DAQIRI_LOG_CRITICAL("ibv_alloc_pd({}) failed: {}", d->name, strerror(errno));
        return false;
      }
      doca_error_t ret = doca_verbs_bridge_verbs_context_import(
          ctx, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE, &d->vctx);
      if (ret == DOCA_SUCCESS) {
        ret = doca_verbs_bridge_verbs_pd_import(d->pd, &d->vpd);
      }
      if (ret == DOCA_SUCCESS) {
        ret = doca_rdma_bridge_open_dev_from_pd(d->pd, &d->dev);
      }
      if (ret != DOCA_SUCCESS) {
        DAQIRI_LOG_CRITICAL("Could not open {} with DOCA Verbs: {}", d->name, doca_err(ret));
        devices_.push_back(std::move(d));
        return false;
      }
      ret = doca_gpu_nic_cap_is_nic_handler_gpu_sm_db_supported(gpu_, d->dev);
      if (ret != DOCA_SUCCESS) {
        DAQIRI_LOG_CRITICAL(
            "CUDA device {} can't ring the doorbells of {}: {}. The gpunetio_rdma engine needs "
            "the GPU to drive the NIC directly (GPU_SM_DB); check that the GPU and the NIC can "
            "use GPUDirect RDMA and that the NVIDIA driver maps the NIC doorbells",
            gpu_ordinal_, d->name, doca_err(ret));
        devices_.push_back(std::move(d));
        return false;
      }
      DAQIRI_LOG_INFO("RDMA device {} opened with DOCA Verbs", d->name);
      device = d.get();
      devices_.push_back(std::move(d));
    }
    if_device_[i] = device;
    DAQIRI_LOG_INFO("Interface '{}' ({}) uses RDMA device {}", intf.name_, intf.address_,
                    device->name);
  }
  return true;
}

namespace {

int mr_access_to_ibv(uint32_t access) {
  int ibv_access = IBV_ACCESS_LOCAL_WRITE;
  if (access & MEM_ACCESS_RDMA_READ) {
    ibv_access |= IBV_ACCESS_REMOTE_READ;
  }
  if (access & MEM_ACCESS_RDMA_WRITE) {
    ibv_access |= IBV_ACCESS_REMOTE_WRITE;
  }
  return ibv_access;
}

// Registers GPU memory through a dma-buf, so that it works without the peermem module
struct ibv_mr* register_device_memory(struct ibv_pd* pd, void* base, size_t size, int access,
                                      CUcontext owning, const std::string& what) {
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
  struct ibv_mr* mr = ibv_reg_dmabuf_mr(pd, offset, size, va, fd, access);
  const int reg_errno = errno;
  close(fd);
  if (mr == nullptr) {
    DAQIRI_LOG_CRITICAL("ibv_reg_dmabuf_mr failed for {} ({} bytes): {}", what, size,
                        strerror(reg_errno));
  }
  return mr;
}

}  // namespace

bool GpunetioRdmaEngine::register_memory_regions() {
  for (auto& device : devices_) {
    for (const auto& [name, mr] : cfg_.mrs_) {
      void* base = ar_[name].ptr_;
      if (base == nullptr) {
        DAQIRI_LOG_CRITICAL("Memory region {} was not allocated", name);
        return false;
      }
      const int access = mr_access_to_ibv(mr.access_);
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
        reg = register_device_memory(device->pd, base, mr.ttl_size_, access,
                                     ar_[name].cuda_context_, "memory region " + name);
      } else {
        reg = ibv_reg_mr(device->pd, base, mr.ttl_size_, access);
        if (reg == nullptr) {
          DAQIRI_LOG_CRITICAL("ibv_reg_mr failed for memory region {} ({} bytes): {}", name,
                              mr.ttl_size_, strerror(errno));
        }
      }
      if (reg == nullptr) {
        return false;
      }
      device->mrs[name] = reg;
      DAQIRI_LOG_INFO("Memory region {} registered on {} ({} bytes, lkey {:#x})", name,
                      device->name, mr.ttl_size_, reg->lkey);
    }

    // A small GPU buffer the memory consistency WQEs of pre-Hopper GPUs read
    if (cudaMalloc(&device->dump_buf, GPU_PAGE_SIZE) != cudaSuccess) {
      device->dump_buf = nullptr;
      DAQIRI_LOG_CRITICAL("Could not allocate the dump buffer of {}", device->name);
      return false;
    }
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    device->dump_mr =
        register_device_memory(device->pd, device->dump_buf, GPU_PAGE_SIZE, IBV_ACCESS_LOCAL_WRITE,
                               ctx, "the dump buffer of " + device->name);
    if (device->dump_mr == nullptr) {
      return false;
    }
  }
  return true;
}

bool GpunetioRdmaEngine::setup_pools() {
  const int numa = daqiri::detail::numa_node_for_cpu(cfg_.common_.master_core_);
  for (const auto& [name, mr] : cfg_.mrs_) {
    // Buffers go back from the workers and from the application threads
    auto* ring = daqiri::Ring::create(("GV_" + name).c_str(), mr.num_bufs_ + 1,
                                      daqiri::RingMode::MPMC, numa);
    if (ring == nullptr) {
      DAQIRI_LOG_CRITICAL("Could not create the buffer pool of memory region {}", name);
      return false;
    }
    buf_pools_[name] = ring;
    if (populate_pool(ring, name) != Status::SUCCESS) {
      return false;
    }
  }
  meta_pool_ = daqiri::ObjectPool::create("GV_META", BURST_POOL, sizeof(BurstParams), numa);
  pkt_ptr_pool_ =
      daqiri::ObjectPool::create("GV_PKT_PTRS", BURST_POOL, sizeof(void*) * MAX_BURST, numa);
  pkt_len_pool_ =
      daqiri::ObjectPool::create("GV_PKT_LENS", BURST_POOL, sizeof(uint32_t) * MAX_BURST, numa);
  if (meta_pool_ == nullptr || pkt_ptr_pool_ == nullptr || pkt_len_pool_ == nullptr) {
    DAQIRI_LOG_CRITICAL("Could not create the burst pools");
    return false;
  }
  return true;
}

bool GpunetioRdmaEngine::create_slots() {
  const int numa = daqiri::detail::numa_node_for_cpu(cfg_.common_.master_core_);
  for (size_t i = 0; i < cfg_.ifs_.size(); i++) {
    const auto& intf = cfg_.ifs_[i];
    const bool server = intf.rdma_.mode_ == RDMAMode::SERVER;
    // Same slot counts as the ibverbs RDMA engine
    const size_t count = server ? intf.rx_.queues_.size() : intf.tx_.queues_.size();
    for (size_t q = 0; q < count; q++) {
      auto slot = std::make_unique<Slot>();
      slot->index = static_cast<int>(slots_.size());
      slot->if_idx = static_cast<int>(i);
      slot->queue_idx = static_cast<int>(q);
      slot->server = server;
      const CommonQueueConfig& qcfg =
          q < intf.tx_.queues_.size() ? intf.tx_.queues_[q].common_ : intf.rx_.queues_[q].common_;
      slot->cpu_core = parse_core(qcfg.cpu_core_);
      slot->name = intf.name_ + "/" + qcfg.name_;
      slot->dev = if_device_[i];

      slot->ctrl = alloc_pinned<gr::SlotCtrl>(1);
      slot->sq_desc = alloc_pinned<gr::BurstDesc>(gr::DESC_RING);
      slot->rq_desc = alloc_pinned<gr::BurstDesc>(gr::DESC_RING);
      slot->sq_pkts = alloc_pinned<gr::Packet>(QP_DEPTH);
      slot->rq_pkts = alloc_pinned<gr::Packet>(QP_DEPTH);
      const std::string ring_name = "GV_SLOT_" + std::to_string(slot->index);
      // One application thread and the worker on each side, as with the ibverbs RDMA engine
      slot->app_tx =
          daqiri::Ring::create((ring_name + "_TX").c_str(), APP_RING, daqiri::RingMode::SPSC, numa);
      slot->app_rx =
          daqiri::Ring::create((ring_name + "_RX").c_str(), APP_RING, daqiri::RingMode::SPSC, numa);
      Slot& ref = *slot;
      slots_.push_back(std::move(slot));
      if (ref.ctrl == nullptr || ref.sq_desc == nullptr || ref.rq_desc == nullptr ||
          ref.sq_pkts == nullptr || ref.rq_pkts == nullptr || ref.app_tx == nullptr ||
          ref.app_rx == nullptr) {
        DAQIRI_LOG_CRITICAL("Could not allocate the rings of slot {}", ref.name);
        return false;
      }
      if (!create_queue_pair(ref)) {
        return false;
      }
      DAQIRI_LOG_INFO("Slot {} ({}): queue pair {:#x} on {}, {} work requests per queue", ref.index,
                      ref.name, ref.qpn, ref.dev->name, QP_DEPTH);
    }
  }
  if (slots_.empty()) {
    DAQIRI_LOG_CRITICAL("No queue slots: interfaces need RX queues (server) or TX queues (client)");
    return false;
  }
  return true;
}

namespace {

// GPU memory for a queue ring, registered with the NIC
bool alloc_gpu_ring(struct doca_gpu* gpu, struct doca_dev* dev, size_t size, void** ptr,
                    struct doca_umem** umem, const char* what) {
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

// Completion queue with its CQEs and doorbell record in GPU memory. Overrun mode: the kernel tracks
// its position itself and never has more work requests in flight than the queue holds.
bool create_cq(struct doca_gpu* gpu, struct doca_dev* dev, struct doca_verbs_context* vctx,
               void** ring, struct doca_umem** umem, struct doca_verbs_cq** cq) {
  const size_t size = round_up(QP_DEPTH * CQE_SIZE + CQ_DBR_SIZE, page_size());
  if (!alloc_gpu_ring(gpu, dev, size, ring, umem, "completion queue")) {
    return false;
  }
  // Every CQE starts invalid and owned by the hardware
  std::vector<struct mlx5_cqe64> cqes(QP_DEPTH);
  std::memset(cqes.data(), 0, cqes.size() * sizeof(cqes[0]));
  for (auto& cqe : cqes) {
    cqe.op_own = static_cast<uint8_t>((MLX5_CQE_INVALID << 4) | MLX5_CQE_OWNER_MASK);
  }
  if (cudaMemcpy(*ring, cqes.data(), cqes.size() * sizeof(cqes[0]), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    DAQIRI_LOG_CRITICAL("Could not initialize a completion queue");
    return false;
  }
  struct doca_verbs_cq_attr* attr = nullptr;
  doca_error_t ret = doca_verbs_cq_attr_create(&attr);
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_attr_set_external_datapath_en(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_attr_set_external_umem(attr, *umem, 0);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_attr_set_cq_size(attr, QP_DEPTH);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_attr_set_cq_overrun(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_cq_create(vctx, attr, cq);
  }
  if (attr != nullptr) {
    doca_verbs_cq_attr_destroy(attr);
  }
  if (ret != DOCA_SUCCESS) {
    *cq = nullptr;
    DAQIRI_LOG_CRITICAL("Could not create a completion queue: {}", doca_err(ret));
    return false;
  }
  return true;
}

}  // namespace

bool GpunetioRdmaEngine::create_queue_pair(Slot& slot) {
  if (!select_cuda_device(gpu_ordinal_, "creating queue pair " + slot.name)) {
    return false;
  }
  Device& d = *slot.dev;
  if (!create_cq(gpu_, d.dev, d.vctx, &slot.cq_sq_ring, &slot.cq_sq_umem, &slot.cq_sq) ||
      !create_cq(gpu_, d.dev, d.vctx, &slot.cq_rq_ring, &slot.cq_rq_umem, &slot.cq_rq)) {
    return false;
  }

  // The kernel rings the doorbells: a non-cached UAR, as the DOCA GPUNetIO samples use
  doca_error_t ret = doca_uar_create(d.dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &slot.uar);
  if (ret != DOCA_SUCCESS) {
    ret = doca_uar_create(d.dev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE, &slot.uar);
  }
  if (ret != DOCA_SUCCESS) {
    ret = doca_uar_create(d.dev, DOCA_UAR_ALLOCATION_TYPE_BLUEFLAME, &slot.uar);
  }
  if (ret != DOCA_SUCCESS) {
    slot.uar = nullptr;
    DAQIRI_LOG_CRITICAL("Could not create a UAR on {}: {}", d.name, doca_err(ret));
    return false;
  }

  const size_t wq_size = round_up(QP_DEPTH * RQ_WQE_SIZE + QP_DEPTH * SQ_WQE_SIZE, page_size());
  if (!alloc_gpu_ring(gpu_, d.dev, wq_size, &slot.qp_ring, &slot.qp_umem, "work queues") ||
      !alloc_gpu_ring(gpu_, d.dev, page_size(), &slot.dbr, &slot.dbr_umem, "doorbell records")) {
    return false;
  }

  struct doca_verbs_qp_init_attr* attr = nullptr;
  ret = doca_verbs_qp_init_attr_create(&attr);
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_external_datapath_en(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_external_uar(attr, slot.uar);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_external_umem(attr, slot.qp_umem, 0);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_external_dbr_umem(attr, slot.dbr_umem, 0);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_pd(attr, d.vpd);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_sq_wr(attr, QP_DEPTH);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_rq_wr(attr, QP_DEPTH);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_qp_type(attr, DOCA_VERBS_QP_TYPE_RC);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_send_cq(attr, slot.cq_sq);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_receive_cq(attr, slot.cq_rq);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_send_max_sges(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_init_attr_set_receive_max_sges(attr, 1);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_create(d.vctx, attr, &slot.qp);
  }
  if (attr != nullptr) {
    doca_verbs_qp_init_attr_destroy(attr);
  }
  if (ret != DOCA_SUCCESS) {
    slot.qp = nullptr;
    DAQIRI_LOG_CRITICAL("Could not create the queue pair of slot {}: {}", slot.name, doca_err(ret));
    return false;
  }
  slot.qpn = doca_verbs_qp_get_qpn(slot.qp);

  ret = doca_gpu_verbs_export_qp(gpu_, d.dev, slot.qp, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                                 slot.qp_ring, slot.cq_sq, slot.cq_rq, &slot.gpu_qp);
  if (ret == DOCA_SUCCESS) {
    ret = doca_gpu_verbs_get_qp_dev(slot.gpu_qp, &slot.dev_qp);
  }
  if (ret != DOCA_SUCCESS) {
    DAQIRI_LOG_CRITICAL("Could not export the queue pair of slot {} to the GPU: {}", slot.name,
                        doca_err(ret));
    return false;
  }
  // On GPUs that need it (before Hopper), the kernel follows each receive burst into GPU memory
  // with a memory consistency WQE on the send queue: keep room for one per receive burst
  bool need_mcst = false;
  if (cudaMemcpy(&need_mcst,
                 reinterpret_cast<const uint8_t*>(slot.dev_qp) +
                     offsetof(struct doca_gpu_dev_verbs_qp, need_mcst),
                 sizeof(need_mcst), cudaMemcpyDeviceToHost) != cudaSuccess) {
    DAQIRI_LOG_CRITICAL("Could not read the GPU queue pair of slot {}", slot.name);
    return false;
  }
  slot.sq_pkt_limit = need_mcst ? QP_DEPTH - gr::DESC_RING : QP_DEPTH;
  return true;
}

void GpunetioRdmaEngine::destroy_queue_pair(Slot& slot) {
  if (slot.gpu_qp != nullptr) {
    doca_gpu_verbs_unexport_qp(gpu_, slot.gpu_qp);
    slot.gpu_qp = nullptr;
    slot.dev_qp = nullptr;
  }
  if (slot.qp != nullptr) {
    doca_verbs_qp_destroy(slot.qp);
    slot.qp = nullptr;
  }
  for (auto* cq : {&slot.cq_sq, &slot.cq_rq}) {
    if (*cq != nullptr) {
      doca_verbs_cq_destroy(*cq);
      *cq = nullptr;
    }
  }
  for (auto* umem : {&slot.qp_umem, &slot.dbr_umem, &slot.cq_sq_umem, &slot.cq_rq_umem}) {
    if (*umem != nullptr) {
      doca_umem_destroy(*umem);
      *umem = nullptr;
    }
  }
  for (auto* ring : {&slot.qp_ring, &slot.dbr, &slot.cq_sq_ring, &slot.cq_rq_ring}) {
    if (*ring != nullptr) {
      doca_gpu_mem_free(gpu_, *ring);
      *ring = nullptr;
    }
  }
  if (slot.uar != nullptr) {
    doca_uar_destroy(slot.uar);
    slot.uar = nullptr;
  }
}

bool GpunetioRdmaEngine::start_kernel() {
  if (!select_cuda_device(gpu_ordinal_, "starting the gpunetio_rdma kernel")) {
    return false;
  }
  // Under CUDA lazy loading, a kernel launched for the first time waits for the running kernels to
  // exit, and this one exits only at shutdown: load it first
  cudaError_t err = gr::load_kernels();
  if (err != cudaSuccess) {
    DAQIRI_LOG_CRITICAL("Could not load the gpunetio_rdma kernel: {}", cudaGetErrorString(err));
    return false;
  }
  const char* loading = std::getenv("CUDA_MODULE_LOADING");
  if (loading == nullptr || std::strcmp(loading, "EAGER") != 0) {
    DAQIRI_LOG_WARN(
        "CUDA lazy loading is on: a kernel the application launches for the first time after "
        "daqiri_init() blocks until shutdown. Set CUDA_MODULE_LOADING=EAGER, or load the "
        "application kernels before daqiri_init() with cudaFuncGetAttributes().");
  }

  auto* args = alloc_pinned<gr::SlotArgs>(slots_.size());
  slot_args_ = args;
  if (args == nullptr ||
      cudaStreamCreateWithFlags(&kernel_stream_, cudaStreamNonBlocking) != cudaSuccess) {
    kernel_stream_ = nullptr;
    DAQIRI_LOG_CRITICAL("Could not allocate the gpunetio_rdma kernel resources");
    return false;
  }
  for (size_t i = 0; i < slots_.size(); i++) {
    const Slot& slot = *slots_[i];
    gr::SlotArgs& a = args[i];
    a.qp = slot.dev_qp;
    a.ctrl = slot.ctrl;
    a.sq_desc = slot.sq_desc;
    a.rq_desc = slot.rq_desc;
    a.sq_pkts = slot.sq_pkts;
    a.rq_pkts = slot.rq_pkts;
    a.sq_pkt_mask = QP_DEPTH_MASK;
    a.rq_pkt_mask = QP_DEPTH_MASK;
    a.dump_addr = reinterpret_cast<uint64_t>(slot.dev->dump_buf);
    a.dump_lkey = htobe32(slot.dev->dump_mr->lkey);
  }
  err = gr::launch_kernel(args, static_cast<uint32_t>(slots_.size()), kernel_stream_);
  if (err != cudaSuccess) {
    DAQIRI_LOG_CRITICAL(
        "Could not start the gpunetio_rdma kernel for {} slot(s): {}. The GPU must run one block "
        "of {} threads per queue slot at once",
        slots_.size(), cudaGetErrorString(err), gr::kernel_threads());
    return false;
  }
  kernel_running_ = true;
  return true;
}

bool GpunetioRdmaEngine::stop_kernel() {
  if (!kernel_running_) {
    return true;
  }
  for (auto& slot : slots_) {
    store_release(&slot->ctrl->exit, 1u);
  }
  const auto t0 = std::chrono::steady_clock::now();
  while (cudaStreamQuery(kernel_stream_) == cudaErrorNotReady) {
    if (std::chrono::steady_clock::now() - t0 > DRAIN_TIMEOUT) {
      DAQIRI_LOG_CRITICAL(
          "The gpunetio_rdma kernel did not stop; its queue resources are not released");
      kernel_stuck_ = true;
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  kernel_running_ = false;
  return true;
}

bool GpunetioRdmaEngine::start_listeners() {
  for (size_t i = 0; i < cfg_.ifs_.size(); i++) {
    const auto& intf = cfg_.ifs_[i];
    if (intf.rdma_.mode_ != RDMAMode::SERVER) {
      continue;
    }
    auto listener = std::make_unique<Listener>();
    listener->if_idx = static_cast<int>(i);
    listener->key = intf.address_ + ":" + std::to_string(intf.rdma_.port_);
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(intf.rdma_.port_);
    inet_pton(AF_INET, intf.address_.c_str(), &addr.sin_addr);
    if (rdma_create_id(cm_channel_, &listener->id, nullptr, RDMA_PS_TCP) != 0) {
      listener->id = nullptr;
      DAQIRI_LOG_CRITICAL("rdma_create_id failed: {}", strerror(errno));
      return false;
    }
    Listener& ref = *listener;
    listeners_.push_back(std::move(listener));
    if (rdma_bind_addr(ref.id, reinterpret_cast<struct sockaddr*>(&addr)) != 0 ||
        rdma_listen(ref.id, static_cast<int>(intf.rx_.queues_.size())) != 0) {
      DAQIRI_LOG_CRITICAL("Could not listen on {}: {}", ref.key, strerror(errno));
      return false;
    }
    DAQIRI_LOG_INFO("RDMA server listening on {} with {} queue slot(s)", ref.key,
                    intf.rx_.queues_.size());
  }
  return true;
}

// ---------------------------------------------------------------------------
// Queue pair state
// ---------------------------------------------------------------------------

// Moves the queue pair to the given ibv state with the attributes rdma_cm computes for the
// connection of id, converted to DOCA Verbs
bool GpunetioRdmaEngine::transition_qp(Slot& slot, struct rdma_cm_id* id, int state) {
  struct ibv_qp_attr attr = {};
  int mask = 0;
  attr.qp_state = static_cast<enum ibv_qp_state>(state);
  if (rdma_init_qp_attr(id, &attr, &mask) != 0) {
    DAQIRI_LOG_ERROR("Slot {}: rdma_init_qp_attr failed: {}", slot.name, strerror(errno));
    return false;
  }
  if (state == IBV_QPS_RTR) {
    attr.min_rnr_timer = static_cast<uint8_t>(env_min_rnr_timer());
    mask |= IBV_QP_MIN_RNR_TIMER;
  }

  struct doca_verbs_qp_attr* qp_attr = nullptr;
  struct doca_verbs_ah_attr* ah_attr = nullptr;
  doca_error_t ret = doca_verbs_qp_attr_create(&qp_attr);
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_ah_attr_create(slot.dev->vctx, &ah_attr);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_attr_set_ah_attr(qp_attr, ah_attr);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_bridge_qp_attr_set(qp_attr, &attr, mask);
  }
  int doca_mask = doca_verbs_bridge_get_qp_attr_mask(mask);
  // The bridge doesn't carry the RDMA read limits
  if (ret == DOCA_SUCCESS && (mask & IBV_QP_MAX_DEST_RD_ATOMIC) != 0) {
    ret = doca_verbs_qp_attr_set_max_dest_rd_atomic(qp_attr, attr.max_dest_rd_atomic);
    doca_mask |= DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC;
  }
  if (ret == DOCA_SUCCESS && (mask & IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
    ret = doca_verbs_qp_attr_set_max_rd_atomic(qp_attr, attr.max_rd_atomic);
    doca_mask |= DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC;
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_modify(slot.qp, qp_attr, doca_mask);
  }
  if (ah_attr != nullptr) {
    doca_verbs_ah_attr_destroy(ah_attr);
  }
  if (qp_attr != nullptr) {
    doca_verbs_qp_attr_destroy(qp_attr);
  }
  if (ret != DOCA_SUCCESS) {
    DAQIRI_LOG_ERROR("Slot {}: could not move queue pair {:#x} to state {}: {}", slot.name,
                     slot.qpn, state, doca_err(ret));
    return false;
  }
  return true;
}

namespace {

bool set_qp_state(struct doca_verbs_qp* qp, enum doca_verbs_qp_state state) {
  struct doca_verbs_qp_attr* attr = nullptr;
  doca_error_t ret = doca_verbs_qp_attr_create(&attr);
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_attr_set_next_state(attr, state);
  }
  if (ret == DOCA_SUCCESS) {
    ret = doca_verbs_qp_modify(qp, attr, DOCA_VERBS_QP_ATTR_NEXT_STATE);
  }
  if (attr != nullptr) {
    doca_verbs_qp_attr_destroy(attr);
  }
  if (ret != DOCA_SUCCESS) {
    DAQIRI_LOG_ERROR("Could not move a queue pair to state {}: {}", static_cast<int>(state),
                     doca_err(ret));
  }
  return ret == DOCA_SUCCESS;
}

}  // namespace

bool GpunetioRdmaEngine::reset_qp(Slot& slot) {
  return set_qp_state(slot.qp, DOCA_VERBS_QP_STATE_RST);
}

bool GpunetioRdmaEngine::set_qp_error(Slot& slot) {
  return set_qp_state(slot.qp, DOCA_VERBS_QP_STATE_ERR);
}

// ---------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------

GpunetioRdmaEngine::Slot* GpunetioRdmaEngine::claim_free_slot(int if_idx, bool server) {
  for (auto& slot : slots_) {
    if (slot->if_idx == if_idx && slot->server == server &&
        slot->host_state.load() == static_cast<uint32_t>(SlotHostState::FREE)) {
      slot->host_state.store(static_cast<uint32_t>(SlotHostState::CONNECTING));
      return slot.get();
    }
  }
  return nullptr;
}

GpunetioRdmaEngine::Slot* GpunetioRdmaEngine::find_slot_by_cm_id(struct rdma_cm_id* id) {
  for (auto& slot : slots_) {
    if (slot->cm_id == id) {
      return slot.get();
    }
  }
  return nullptr;
}

GpunetioRdmaEngine::Slot* GpunetioRdmaEngine::find_slot(uintptr_t conn_id) {
  // Connection IDs are (generation << 32) | (slot index + 1), so a stale ID never matches a later
  // connection of the same slot
  const uint64_t index = (conn_id & 0xffffffffULL) - 1;
  if (conn_id == 0 || index >= slots_.size()) {
    return nullptr;
  }
  return slots_[index].get();
}

// Makes a connected slot usable by the application and the kernel
void GpunetioRdmaEngine::activate_slot(Slot& slot) {
  std::lock_guard<std::mutex> lock(slots_mutex_);
  slot.generation++;
  slot.conn_id.store((static_cast<uintptr_t>(slot.generation) << 32) |
                     static_cast<uintptr_t>(slot.index + 1));
  slot.claimed = false;
  slot.close_requested.store(false);
  store_release(&slot.ctrl->state, static_cast<uint32_t>(gr::SLOT_ACTIVE));
  slot.connections.fetch_add(1);
  slot.host_state.store(static_cast<uint32_t>(SlotHostState::ACTIVE));
  DAQIRI_LOG_INFO("Slot {}: {} connection {:#x} established on queue pair {:#x}", slot.name,
                  slot.server ? "server" : "client", slot.conn_id.load(), slot.qpn);
}

// Gives back a slot whose connection failed before it became active
void GpunetioRdmaEngine::release_connecting_slot(Slot& slot) {
  reset_qp(slot);
  std::lock_guard<std::mutex> lock(slots_mutex_);
  slot.cm_id = nullptr;
  slot.host_state.store(static_cast<uint32_t>(SlotHostState::FREE));
}

void GpunetioRdmaEngine::handle_connect_request(struct rdma_cm_event* event) {
  Listener* listener = nullptr;
  for (auto& l : listeners_) {
    if (l->id == event->listen_id) {
      listener = l.get();
    }
  }
  Slot* slot = nullptr;
  if (listener != nullptr) {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    slot = claim_free_slot(listener->if_idx, true);
    if (slot != nullptr) {
      slot->cm_id = event->id;
    }
  }
  if (slot == nullptr) {
    DAQIRI_LOG_WARN("Rejecting a connection request: no free queue slot on {}",
                    listener != nullptr ? listener->key : std::string("an unknown listener"));
    rdma_reject(event->id, nullptr, 0);
    return;
  }
  if (event->id->verbs != slot->dev->ctx) {
    DAQIRI_LOG_ERROR("Slot {}: connection request on another RDMA device", slot->name);
    release_connecting_slot(*slot);
    rdma_reject(event->id, nullptr, 0);
    return;
  }
  // rdma_cm moves the queue pair to RTS before it accepts, the engine does the same with its own
  if (!transition_qp(*slot, event->id, IBV_QPS_INIT) ||
      !transition_qp(*slot, event->id, IBV_QPS_RTR) ||
      !transition_qp(*slot, event->id, IBV_QPS_RTS)) {
    release_connecting_slot(*slot);
    rdma_reject(event->id, nullptr, 0);
    return;
  }
  struct rdma_conn_param param = {};
  param.qp_num = slot->qpn;
  param.responder_resources = RDMA_READ_DEPTH;
  param.initiator_depth = RDMA_READ_DEPTH;
  param.rnr_retry_count = 7;
  if (rdma_accept(event->id, &param) != 0) {
    DAQIRI_LOG_ERROR("Slot {}: rdma_accept failed: {}", slot->name, strerror(errno));
    release_connecting_slot(*slot);
    rdma_reject(event->id, nullptr, 0);
    return;
  }
  DAQIRI_LOG_INFO("Slot {}: accepted a connection request", slot->name);
}

void GpunetioRdmaEngine::handle_established(struct rdma_cm_id* id) {
  Slot* slot = nullptr;
  {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    slot = find_slot_by_cm_id(id);
  }
  if (slot == nullptr ||
      slot->host_state.load() != static_cast<uint32_t>(SlotHostState::CONNECTING)) {
    return;
  }
  activate_slot(*slot);
}

// Returns true when the CM thread must destroy the ID once it acknowledged the event
bool GpunetioRdmaEngine::handle_disconnect(struct rdma_cm_id* id) {
  Slot* slot = nullptr;
  {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    slot = find_slot_by_cm_id(id);
  }
  if (slot == nullptr) {
    return false;
  }
  const auto state = static_cast<SlotHostState>(slot->host_state.load());
  if (state == SlotHostState::ACTIVE) {
    // The worker drains the slot and destroys the ID
    DAQIRI_LOG_INFO("Slot {}: connection {:#x} closed by the peer", slot->name,
                    slot->conn_id.load());
    slot->close_requested.store(true);
  } else if (state == SlotHostState::CONNECTING && slot->server) {
    // The connection never became active: give the slot back
    release_connecting_slot(*slot);
    return true;
  }
  return false;
}

// Connection manager thread: accepts the clients of the server interfaces and tracks disconnects
void GpunetioRdmaEngine::run() {
  pin_thread(cfg_.common_.master_core_, "RDMA connection manager");
  while (cm_running_.load()) {
    struct rdma_cm_event* event = nullptr;
    if (rdma_get_cm_event(cm_channel_, &event) != 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    }
    struct rdma_cm_id* destroy = nullptr;
    switch (event->event) {
      case RDMA_CM_EVENT_CONNECT_REQUEST: {
        handle_connect_request(event);
        Slot* slot = nullptr;
        {
          std::lock_guard<std::mutex> lock(slots_mutex_);
          slot = find_slot_by_cm_id(event->id);
        }
        if (slot == nullptr) {
          destroy = event->id;  // rejected
        }
        break;
      }
      case RDMA_CM_EVENT_ESTABLISHED:
        handle_established(event->id);
        break;
      case RDMA_CM_EVENT_DISCONNECTED:
      case RDMA_CM_EVENT_DEVICE_REMOVAL:
        if (handle_disconnect(event->id)) {
          destroy = event->id;
        }
        break;
      case RDMA_CM_EVENT_REJECTED:
      case RDMA_CM_EVENT_UNREACHABLE:
      case RDMA_CM_EVENT_CONNECT_ERROR: {
        Slot* slot = nullptr;
        {
          std::lock_guard<std::mutex> lock(slots_mutex_);
          slot = find_slot_by_cm_id(event->id);
        }
        if (slot != nullptr && slot->server &&
            slot->host_state.load() == static_cast<uint32_t>(SlotHostState::CONNECTING)) {
          DAQIRI_LOG_WARN("Slot {}: connection failed: {}", slot->name,
                          rdma_event_str(event->event));
          release_connecting_slot(*slot);
          destroy = event->id;
        }
        break;
      }
      default:
        DAQIRI_LOG_DEBUG("Ignoring RDMA CM event {}", rdma_event_str(event->event));
        break;
    }
    rdma_ack_cm_event(event);
    if (destroy != nullptr) {
      rdma_destroy_id(destroy);
    }
  }
}

namespace {

// Waits for the next event on a synchronous client channel
bool wait_cm_event(struct rdma_event_channel* channel, enum rdma_cm_event_type expected,
                   const char* what) {
  struct rdma_cm_event* event = nullptr;
  if (rdma_get_cm_event(channel, &event) != 0) {
    DAQIRI_LOG_ERROR("rdma_get_cm_event failed while waiting for {}: {}", what, strerror(errno));
    return false;
  }
  const bool ok = event->event == expected && event->status == 0;
  if (!ok) {
    DAQIRI_LOG_ERROR("{} failed: {} (status {})", what, rdma_event_str(event->event),
                     event->status);
  }
  rdma_ack_cm_event(event);
  return ok;
}

}  // namespace

Status GpunetioRdmaEngine::rdma_connect_to_server(const std::string& dst_addr, uint16_t dst_port,
                                                  uintptr_t* conn_id) {
  return rdma_connect_to_server(dst_addr, dst_port, "", conn_id);
}

Status GpunetioRdmaEngine::rdma_connect_to_server(const std::string& dst_addr, uint16_t dst_port,
                                                  const std::string& src_addr, uintptr_t* conn_id) {
  if (conn_id == nullptr) {
    return Status::NULL_PTR;
  }
  *conn_id = 0;
  if (!initialized_) {
    return Status::NOT_READY;
  }
  struct sockaddr_in dst = {};
  dst.sin_family = AF_INET;
  dst.sin_port = htons(dst_port);
  if (inet_pton(AF_INET, dst_addr.c_str(), &dst.sin_addr) != 1) {
    DAQIRI_LOG_ERROR("Invalid server address {}", dst_addr);
    return Status::INVALID_PARAMETER;
  }
  struct sockaddr_in src = {};
  src.sin_family = AF_INET;
  if (!src_addr.empty() && inet_pton(AF_INET, src_addr.c_str(), &src.sin_addr) != 1) {
    DAQIRI_LOG_ERROR("Invalid source address {}", src_addr);
    return Status::INVALID_PARAMETER;
  }

  // The connection is set up on a private channel, then moved to the engine's channel so that the
  // connection manager sees its disconnect
  struct rdma_event_channel* channel = rdma_create_event_channel();
  struct rdma_cm_id* id = nullptr;
  if (channel == nullptr || rdma_create_id(channel, &id, nullptr, RDMA_PS_TCP) != 0) {
    DAQIRI_LOG_ERROR("Could not create an RDMA CM ID: {}", strerror(errno));
    if (channel != nullptr) {
      rdma_destroy_event_channel(channel);
    }
    return Status::CONNECT_FAILURE;
  }
  Slot* slot = nullptr;
  const auto fail = [&](Status status) {
    if (slot != nullptr) {
      release_connecting_slot(*slot);
    }
    rdma_destroy_id(id);
    rdma_destroy_event_channel(channel);
    return status;
  };

  if (rdma_resolve_addr(id, src_addr.empty() ? nullptr : reinterpret_cast<struct sockaddr*>(&src),
                        reinterpret_cast<struct sockaddr*>(&dst), CM_TIMEOUT_MS) != 0 ||
      !wait_cm_event(channel, RDMA_CM_EVENT_ADDR_RESOLVED, "Address resolution")) {
    return fail(Status::CONNECT_FAILURE);
  }
  if (rdma_resolve_route(id, CM_TIMEOUT_MS) != 0 ||
      !wait_cm_event(channel, RDMA_CM_EVENT_ROUTE_RESOLVED, "Route resolution")) {
    return fail(Status::CONNECT_FAILURE);
  }

  // The source address picks the client interface
  char local[INET_ADDRSTRLEN] = {};
  const auto* local_addr = reinterpret_cast<struct sockaddr_in*>(rdma_get_local_addr(id));
  inet_ntop(AF_INET, &local_addr->sin_addr, local, sizeof(local));
  int if_idx = -1;
  for (size_t i = 0; i < cfg_.ifs_.size(); i++) {
    if (cfg_.ifs_[i].address_ == local && cfg_.ifs_[i].rdma_.mode_ == RDMAMode::CLIENT) {
      if_idx = static_cast<int>(i);
      break;
    }
  }
  if (if_idx < 0) {
    DAQIRI_LOG_ERROR("No client interface has the source address {}", local);
    return fail(Status::INVALID_PARAMETER);
  }
  {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    slot = claim_free_slot(if_idx, false);
    if (slot != nullptr) {
      slot->cm_id = id;
    }
  }
  if (slot == nullptr) {
    DAQIRI_LOG_ERROR("No free client queue slot on interface '{}'", cfg_.ifs_[if_idx].name_);
    return fail(Status::NO_SPACE_AVAILABLE);
  }
  if (id->verbs != slot->dev->ctx) {
    DAQIRI_LOG_ERROR("Slot {}: the route to {} leaves from another RDMA device", slot->name,
                     dst_addr);
    return fail(Status::CONNECT_FAILURE);
  }

  if (!transition_qp(*slot, id, IBV_QPS_INIT)) {
    return fail(Status::CONNECT_FAILURE);
  }
  struct rdma_conn_param param = {};
  param.qp_num = slot->qpn;
  param.responder_resources = RDMA_READ_DEPTH;
  param.initiator_depth = RDMA_READ_DEPTH;
  param.retry_count = 7;
  param.rnr_retry_count = 7;
  if (rdma_connect(id, &param) != 0) {
    DAQIRI_LOG_ERROR("Slot {}: rdma_connect to {}:{} failed: {}", slot->name, dst_addr, dst_port,
                     strerror(errno));
    return fail(Status::CONNECT_FAILURE);
  }
  // Without a queue pair attached to the CM ID, rdma_cm reports the server reply and the engine
  // completes the connection itself
  if (!wait_cm_event(channel, RDMA_CM_EVENT_CONNECT_RESPONSE, "Connection") ||
      !transition_qp(*slot, id, IBV_QPS_RTR) || !transition_qp(*slot, id, IBV_QPS_RTS)) {
    return fail(Status::CONNECT_FAILURE);
  }
  if (rdma_establish(id) != 0) {
    DAQIRI_LOG_ERROR("Slot {}: rdma_establish failed: {}", slot->name, strerror(errno));
    return fail(Status::CONNECT_FAILURE);
  }
  if (rdma_migrate_id(id, cm_channel_) != 0) {
    DAQIRI_LOG_WARN("Slot {}: could not move the connection to the engine's CM channel: {}",
                    slot->name, strerror(errno));
  } else {
    rdma_destroy_event_channel(channel);
  }
  activate_slot(*slot);
  *conn_id = slot->conn_id.load();
  DAQIRI_LOG_INFO("Connected to server {}:{} from {}", dst_addr, dst_port, local);
  return Status::SUCCESS;
}

Status GpunetioRdmaEngine::rdma_get_server_conn_id(const std::string& server_addr,
                                                   uint16_t server_port, uintptr_t* conn_id) {
  if (conn_id == nullptr) {
    return Status::NULL_PTR;
  }
  const std::string key = server_addr + ":" + std::to_string(server_port);
  const Listener* listener = nullptr;
  for (const auto& l : listeners_) {
    if (l->key == key) {
      listener = l.get();
    }
  }
  if (listener == nullptr) {
    return Status::NOT_READY;
  }
  std::lock_guard<std::mutex> lock(slots_mutex_);
  bool any_connected = false;
  for (auto& slot : slots_) {
    if (slot->if_idx != listener->if_idx || !slot->server ||
        slot->host_state.load() != static_cast<uint32_t>(SlotHostState::ACTIVE)) {
      continue;
    }
    any_connected = true;
    if (!slot->claimed) {
      slot->claimed = true;
      *conn_id = slot->conn_id.load();
      return Status::SUCCESS;
    }
  }
  // Every slot connected and claimed: the application must close a connection first
  bool all_claimed = any_connected;
  for (auto& slot : slots_) {
    if (slot->if_idx == listener->if_idx &&
        slot->host_state.load() != static_cast<uint32_t>(SlotHostState::ACTIVE)) {
      all_claimed = false;
    }
  }
  return all_claimed ? Status::NO_SPACE_AVAILABLE : Status::NOT_READY;
}

Status GpunetioRdmaEngine::rdma_get_port_queue(uintptr_t conn_id, uint16_t* port, uint16_t* queue) {
  Slot* slot = find_slot(conn_id);
  if (slot == nullptr || slot->conn_id.load() != conn_id || port == nullptr || queue == nullptr) {
    return Status::INVALID_PARAMETER;
  }
  *port = static_cast<uint16_t>(slot->if_idx);
  *queue = static_cast<uint16_t>(slot->queue_idx);
  return Status::SUCCESS;
}

Status GpunetioRdmaEngine::rdma_set_header(BurstParams* burst, RDMAOpCode op_code,
                                           uintptr_t conn_id, bool is_server, int num_pkts,
                                           uint64_t wr_id, const std::string& local_mr_name) {
  if (burst == nullptr) {
    return Status::NULL_PTR;
  }
  if (num_pkts < 0 || static_cast<uint32_t>(num_pkts) > MAX_BURST) {
    DAQIRI_LOG_ERROR("A burst holds at most {} messages", MAX_BURST);
    return Status::INVALID_PARAMETER;
  }
  burst->transport_hdr.opcode = op_code;
  set_connection_id(burst, conn_id);
  burst->transport_hdr.server = is_server;
  burst->transport_hdr.num_pkts = static_cast<size_t>(num_pkts);
  burst->transport_hdr.num_segs = 1;
  burst->transport_hdr.wr_id = wr_id;
  snprintf(burst->transport_hdr.local_mr_name, sizeof(burst->transport_hdr.local_mr_name), "%s",
           local_mr_name.c_str());
  return Status::SUCCESS;
}

RDMAOpCode GpunetioRdmaEngine::rdma_get_opcode(BurstParams* burst) {
  return burst != nullptr ? burst->transport_hdr.opcode : RDMAOpCode::INVALID;
}

// ---------------------------------------------------------------------------
// Bursts
// ---------------------------------------------------------------------------

BurstParams* GpunetioRdmaEngine::create_tx_burst_params() {
  BurstParams* burst = nullptr;
  if (meta_pool_ == nullptr || !meta_pool_->get(reinterpret_cast<void**>(&burst))) {
    DAQIRI_LOG_ERROR("No free burst descriptor");
    return nullptr;
  }
  reset_burst_metadata(burst);
  return burst;
}

bool GpunetioRdmaEngine::is_tx_burst_available(BurstParams* burst) {
  if (burst == nullptr) {
    return false;
  }
  const auto pool = buf_pools_.find(burst->transport_hdr.local_mr_name);
  return pool != buf_pools_.end() && pool->second->count() >= burst->transport_hdr.num_pkts;
}

Status GpunetioRdmaEngine::get_tx_packet_burst(BurstParams* burst) {
  if (burst == nullptr) {
    return Status::INVALID_PARAMETER;
  }
  const uint32_t num = static_cast<uint32_t>(burst->transport_hdr.num_pkts);
  if (num > MAX_BURST) {
    return Status::INVALID_PARAMETER;
  }
  const auto pool = buf_pools_.find(burst->transport_hdr.local_mr_name);
  if (pool == buf_pools_.end()) {
    DAQIRI_LOG_ERROR("Unknown memory region {}", burst->transport_hdr.local_mr_name);
    return Status::INVALID_PARAMETER;
  }
  if (!pkt_ptr_pool_->get(reinterpret_cast<void**>(&burst->pkts[0]))) {
    burst->pkts[0] = nullptr;
    return Status::NO_FREE_BURST_BUFFERS;
  }
  if (pool->second->dequeue_bulk(reinterpret_cast<void**>(burst->pkts[0]), num) != num) {
    pkt_ptr_pool_->put(burst->pkts[0]);
    burst->pkts[0] = nullptr;
    return Status::NO_FREE_BURST_BUFFERS;
  }
  if (!pkt_len_pool_->get(reinterpret_cast<void**>(&burst->pkt_lens[0]))) {
    pool->second->enqueue_bulk(reinterpret_cast<void**>(burst->pkts[0]), num);
    pkt_ptr_pool_->put(burst->pkts[0]);
    burst->pkts[0] = nullptr;
    burst->pkt_lens[0] = nullptr;
    return Status::NO_FREE_PACKET_BUFFERS;
  }
  burst->transport_hdr.num_segs = 1;
  return Status::SUCCESS;
}

Status GpunetioRdmaEngine::set_packet_lengths(BurstParams* burst, int idx,
                                              const std::initializer_list<int>& lens) {
  if (burst == nullptr || lens.size() != 1 || burst->pkt_lens[0] == nullptr) {
    return Status::INVALID_PARAMETER;
  }
  burst->pkt_lens[0][idx] = static_cast<uint32_t>(*lens.begin());
  return Status::SUCCESS;
}

Status GpunetioRdmaEngine::set_packet_tx_time(BurstParams* burst, int idx, uint64_t time) {
  (void)burst;
  (void)idx;
  (void)time;
  return Status::NOT_SUPPORTED;
}

Status GpunetioRdmaEngine::set_eth_header(BurstParams* burst, int idx, char* dst_addr) {
  (void)burst;
  (void)idx;
  (void)dst_addr;
  DAQIRI_LOG_ERROR("Cannot set an Ethernet header on an RDMA connection");
  return Status::NOT_SUPPORTED;
}

Status GpunetioRdmaEngine::set_ipv4_header(BurstParams* burst, int idx, int ip_len, uint8_t proto,
                                           unsigned int src_host, unsigned int dst_host) {
  (void)burst;
  (void)idx;
  (void)ip_len;
  (void)proto;
  (void)src_host;
  (void)dst_host;
  DAQIRI_LOG_ERROR("Cannot set an IPv4 header on an RDMA connection");
  return Status::NOT_SUPPORTED;
}

Status GpunetioRdmaEngine::set_udp_header(BurstParams* burst, int idx, int udp_len,
                                          uint16_t src_port, uint16_t dst_port) {
  (void)burst;
  (void)idx;
  (void)udp_len;
  (void)src_port;
  (void)dst_port;
  DAQIRI_LOG_ERROR("Cannot set a UDP header on an RDMA connection");
  return Status::NOT_SUPPORTED;
}

// Copies the message into its buffer, which may be in GPU memory
Status GpunetioRdmaEngine::set_udp_payload(BurstParams* burst, int idx, void* data, int len) {
  if (burst == nullptr || burst->pkts[0] == nullptr || len < 0) {
    return Status::INVALID_PARAMETER;
  }
  return cudaMemcpy(burst->pkts[0][idx], data, static_cast<size_t>(len), cudaMemcpyDefault) ==
                 cudaSuccess
             ? Status::SUCCESS
             : Status::GENERIC_FAILURE;
}

Status GpunetioRdmaEngine::send_tx_burst(BurstParams* burst) {
  if (burst == nullptr) {
    return Status::INVALID_PARAMETER;
  }
  const uintptr_t conn_id = get_connection_id(burst);
  Slot* slot = find_slot(conn_id);
  if (slot == nullptr) {
    DAQIRI_LOG_ERROR("Invalid connection ID {:#x} in send_tx_burst", conn_id);
    free_tx_burst(burst);
    return Status::INVALID_PARAMETER;
  }
  // The worker closes the slot only once no application call uses its rings
  slot->app_users.fetch_add(1);
  Status status = Status::SUCCESS;
  if (slot->host_state.load() != static_cast<uint32_t>(SlotHostState::ACTIVE) ||
      slot->conn_id.load() != conn_id) {
    status = Status::INVALID_PARAMETER;
  } else if (!slot->app_tx->enqueue(burst)) {
    status = Status::NO_SPACE_AVAILABLE;
  }
  slot->app_users.fetch_sub(1);
  if (status != Status::SUCCESS) {
    free_tx_burst(burst);
  }
  return status;
}

Status GpunetioRdmaEngine::get_rx_burst(BurstParams** burst, int port, int q) {
  (void)burst;
  (void)port;
  (void)q;
  return Status::NOT_SUPPORTED;
}

Status GpunetioRdmaEngine::get_rx_burst(BurstParams** burst, uintptr_t conn_id, bool server) {
  (void)server;
  if (burst == nullptr) {
    return Status::NULL_PTR;
  }
  Slot* slot = find_slot(conn_id);
  if (slot == nullptr) {
    return Status::NOT_READY;
  }
  slot->app_users.fetch_add(1);
  Status status = Status::NOT_READY;
  if (slot->host_state.load() == static_cast<uint32_t>(SlotHostState::ACTIVE) &&
      slot->conn_id.load() == conn_id && slot->app_rx->dequeue(reinterpret_cast<void**>(burst))) {
    status = Status::SUCCESS;
  }
  slot->app_users.fetch_sub(1);
  return status;
}

void GpunetioRdmaEngine::free_tx_burst(BurstParams* burst) {
  if (burst == nullptr) {
    return;
  }
  const auto pool = buf_pools_.find(burst->transport_hdr.local_mr_name);
  if (burst->pkts[0] != nullptr) {
    if (pool != buf_pools_.end() && burst->transport_hdr.num_pkts > 0) {
      const auto num = static_cast<unsigned>(burst->transport_hdr.num_pkts);
      if (pool->second->enqueue_bulk(reinterpret_cast<void**>(burst->pkts[0]), num) != num) {
        DAQIRI_LOG_ERROR("Could not give {} buffers back to memory region {}", num,
                         burst->transport_hdr.local_mr_name);
      }
    }
    pkt_ptr_pool_->put(burst->pkts[0]);
  }
  if (burst->pkt_lens[0] != nullptr) {
    pkt_len_pool_->put(burst->pkt_lens[0]);
  }
  reset_burst_metadata(burst);
  meta_pool_->put(burst);
}

void GpunetioRdmaEngine::free_tx_metadata(BurstParams* burst) {
  if (burst != nullptr && meta_pool_ != nullptr) {
    reset_burst_metadata(burst);
    meta_pool_->put(burst);
  }
}

void GpunetioRdmaEngine::free_rx_metadata(BurstParams* burst) {
  free_tx_metadata(burst);
}

void GpunetioRdmaEngine::free_rx_burst(BurstParams* burst) {
  free_tx_metadata(burst);
}

void GpunetioRdmaEngine::free_all_segment_packets(BurstParams* burst, int seg) {
  (void)burst;
  (void)seg;
}

void GpunetioRdmaEngine::free_all_packets(BurstParams* burst) {
  (void)burst;
}

void GpunetioRdmaEngine::free_packet_segment(BurstParams* burst, int seg, int pkt) {
  (void)burst;
  (void)seg;
  (void)pkt;
}

void GpunetioRdmaEngine::free_packet(BurstParams* burst, int pkt) {
  (void)burst;
  (void)pkt;
}

void* GpunetioRdmaEngine::get_packet_ptr(BurstParams* burst, int idx) {
  return burst->pkts[0][idx];
}

uint32_t GpunetioRdmaEngine::get_packet_length(BurstParams* burst, int idx) {
  return burst->pkt_lens[0][idx];
}

void* GpunetioRdmaEngine::get_segment_packet_ptr(BurstParams* burst, int seg, int idx) {
  return burst->pkts[seg][idx];
}

uint32_t GpunetioRdmaEngine::get_segment_packet_length(BurstParams* burst, int seg, int idx) {
  return burst->pkt_lens[seg][idx];
}

FlowId GpunetioRdmaEngine::get_packet_flow_id(BurstParams* burst, int idx) {
  (void)burst;
  (void)idx;
  return 0;
}

Status GpunetioRdmaEngine::get_packet_rx_timestamp(BurstParams* burst, int idx,
                                                   uint64_t* timestamp_ns) {
  (void)burst;
  (void)idx;
  (void)timestamp_ns;
  return Status::NOT_SUPPORTED;
}

void* GpunetioRdmaEngine::get_packet_extra_info(BurstParams* burst, int idx) {
  (void)burst;
  (void)idx;
  return nullptr;
}

uint64_t GpunetioRdmaEngine::get_burst_tot_byte(BurstParams* burst) {
  if (burst == nullptr || burst->pkt_lens[0] == nullptr) {
    return 0;
  }
  uint64_t total = 0;
  for (size_t i = 0; i < burst->transport_hdr.num_pkts; i++) {
    total += burst->pkt_lens[0][i];
  }
  return total;
}

Status GpunetioRdmaEngine::get_mac_addr(int port, char* mac) {
  (void)port;
  (void)mac;
  return Status::NOT_SUPPORTED;
}

// ---------------------------------------------------------------------------
// Slot workers
// ---------------------------------------------------------------------------

uint32_t GpunetioRdmaEngine::lkey_for(const Device& dev, const std::string& mr_name) const {
  const auto mr = dev.mrs.find(mr_name);
  return mr != dev.mrs.end() ? mr->second->lkey : 0;
}

// Returns a finished burst to the application, in completion order
void GpunetioRdmaEngine::complete(Slot& slot, BurstParams* burst, Status status) {
  if (status != Status::SUCCESS) {
    slot.failed_bursts.fetch_add(1);
  }
  if (slot.closing) {
    // The connection is gone: nobody can collect the burst
    free_tx_burst(burst);
    slot.closing_freed++;
    return;
  }
  burst->transport_hdr.status = status;
  burst->transport_hdr.tx = burst->transport_hdr.opcode != RDMAOpCode::RECEIVE;
  burst->transport_hdr.server = slot.server;
  if (!slot.completions.empty() || !slot.app_rx->enqueue(burst)) {
    slot.completions.push_back(burst);
  }
}

void GpunetioRdmaEngine::flush_completions(Slot& slot) {
  while (!slot.completions.empty() && slot.app_rx->enqueue(slot.completions.front())) {
    slot.completions.pop_front();
  }
}

// Puts a burst on the GPU ring of its direction. Returns false when the ring is full; the worker
// retries later.
bool GpunetioRdmaEngine::publish(Slot& slot, BurstParams* burst) {
  const RDMAOpCode op = burst->transport_hdr.opcode;
  if (op != RDMAOpCode::SEND && op != RDMAOpCode::RECEIVE) {
    // One-sided operations need remote keys the API can't exchange yet
    complete(slot, burst, Status::NOT_SUPPORTED);
    return true;
  }
  const uint32_t num = static_cast<uint32_t>(burst->transport_hdr.num_pkts);
  if (num == 0) {
    complete(slot, burst, Status::SUCCESS);
    return true;
  }
  const bool send = op == RDMAOpCode::SEND;
  const uint32_t limit = send ? slot.sq_pkt_limit : QP_DEPTH;
  const std::string mr_name = burst->transport_hdr.local_mr_name;
  // The DOCA WQE functions take the memory keys in network byte order
  const uint32_t lkey = htobe32(lkey_for(*slot.dev, mr_name));
  bool valid = num <= limit && burst->pkts[0] != nullptr && burst->pkt_lens[0] != nullptr &&
               slot.dev->mrs.count(mr_name) != 0;
  for (uint32_t i = 0; valid && i < num; i++) {
    valid = burst->pkt_lens[0][i] <= MAX_MESSAGE_SIZE;
  }
  if (!valid) {
    DAQIRI_LOG_ERROR(
        "Slot {}: invalid burst ({} messages, memory region '{}'): at most {} messages of at "
        "most {} bytes from a registered memory region",
        slot.name, num, mr_name, limit, MAX_MESSAGE_SIZE);
    complete(slot, burst, Status::INVALID_PARAMETER);
    return true;
  }

  auto& inflight = send ? slot.sq_inflight : slot.rq_inflight;
  uint32_t& pkts_inflight = send ? slot.sq_pkts_inflight : slot.rq_pkts_inflight;
  if (inflight.size() >= gr::DESC_RING || pkts_inflight + num > limit) {
    return false;
  }
  gr::Packet* pkts = send ? slot.sq_pkts : slot.rq_pkts;
  uint32_t& head = send ? slot.sq_pkt_head : slot.rq_pkt_head;
  for (uint32_t i = 0; i < num; i++) {
    gr::Packet& pkt = pkts[(head + i) & QP_DEPTH_MASK];
    pkt.addr = reinterpret_cast<uint64_t>(burst->pkts[0][i]);
    pkt.len = burst->pkt_lens[0][i];
    pkt.lkey = lkey;
  }
  uint64_t& published = send ? slot.sq_published : slot.rq_published;
  gr::BurstDesc& desc = (send ? slot.sq_desc : slot.rq_desc)[published & (gr::DESC_RING - 1)];
  desc.first_pkt = head;
  desc.num_pkts = num;
  desc.flags = 0;
  if (!send) {
    const auto mr = cfg_.mrs_.find(mr_name);
    if (mr != cfg_.mrs_.end() && mr->second.kind_ == MemoryKind::DEVICE) {
      desc.flags |= gr::DESC_FLAG_GPU_BUFFERS;
    }
  }
  desc.status = gr::BURST_OK;
  store_release(&desc.ready, published + 1);
  inflight.push_back({burst, head, num});
  head += num;
  pkts_inflight += num;
  published++;
  return true;
}

// Collects the bursts the kernel completed, in order
void GpunetioRdmaEngine::reap(Slot& slot) {
  const uint64_t sq_done = load_acquire(&slot.ctrl->sq_done);
  while (slot.sq_reaped < sq_done && !slot.sq_inflight.empty()) {
    const Slot::Inflight in = slot.sq_inflight.front();
    slot.sq_inflight.pop_front();
    const uint32_t status = slot.sq_desc[slot.sq_reaped & (gr::DESC_RING - 1)].status;
    slot.sq_reaped++;
    slot.sq_pkts_inflight -= in.num_pkts;
    if (status == gr::BURST_OK) {
      uint64_t bytes = 0;
      for (uint32_t i = 0; i < in.num_pkts; i++) {
        bytes += in.burst->pkt_lens[0][i];
      }
      slot.send_msgs.fetch_add(in.num_pkts, std::memory_order_relaxed);
      slot.send_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    complete(slot, in.burst, status == gr::BURST_OK ? Status::SUCCESS : Status::GENERIC_FAILURE);
  }

  const uint64_t rq_done = load_acquire(&slot.ctrl->rq_done);
  while (slot.rq_reaped < rq_done && !slot.rq_inflight.empty()) {
    const Slot::Inflight in = slot.rq_inflight.front();
    slot.rq_inflight.pop_front();
    const uint32_t status = slot.rq_desc[slot.rq_reaped & (gr::DESC_RING - 1)].status;
    slot.rq_reaped++;
    slot.rq_pkts_inflight -= in.num_pkts;
    if (status == gr::BURST_OK) {
      // The received byte counts replace the buffer sizes
      uint64_t bytes = 0;
      for (uint32_t i = 0; i < in.num_pkts; i++) {
        const uint32_t len = slot.rq_pkts[(in.first_pkt + i) & QP_DEPTH_MASK].len;
        in.burst->pkt_lens[0][i] = len;
        bytes += len;
      }
      slot.recv_msgs.fetch_add(in.num_pkts, std::memory_order_relaxed);
      slot.recv_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    complete(slot, in.burst, status == gr::BURST_OK ? Status::SUCCESS : Status::GENERIC_FAILURE);
  }

  const uint32_t error = load_acquire(&slot.ctrl->error);
  if (error != 0 && error != slot.logged_error) {
    slot.logged_error = error;
    const uint32_t syndrome = (error >> 8) & 0xff;
    DAQIRI_LOG_ERROR("Slot {}: {} completion error: {} (syndrome {:#x}, vendor syndrome {:#x})",
                     slot.name, (error >> 16) == gr::ERROR_SEND ? "send" : "receive",
                     syndrome_str(syndrome), syndrome, error & 0xff);
  }
}

void GpunetioRdmaEngine::free_burst_internal(BurstParams* burst) {
  if (burst != nullptr) {
    free_tx_burst(burst);
  }
}

// Worker thread side of a disconnect: stops the application calls on the slot, flushes the queue
// pair through the kernel, frees the bursts nobody can collect any more and resets the queue pair
// for the next connection
void GpunetioRdmaEngine::close_slot(Slot& slot) {
  DAQIRI_LOG_INFO("Slot {}: closing connection {:#x}", slot.name, slot.conn_id.load());
  slot.host_state.store(static_cast<uint32_t>(SlotHostState::CLOSING));
  while (slot.app_users.load() != 0) {
    cpu_relax();
  }
  slot.closing = true;
  slot.closing_freed = 0;

  // Every posted work request completes once the queue pair is in the error state
  set_qp_error(slot);
  store_release(&slot.ctrl->state, static_cast<uint32_t>(gr::SLOT_CLOSING));
  const auto t0 = std::chrono::steady_clock::now();
  while (load_acquire(&slot.ctrl->state) != static_cast<uint32_t>(gr::SLOT_DRAINED)) {
    reap(slot);
    if (std::chrono::steady_clock::now() - t0 > DRAIN_TIMEOUT) {
      DAQIRI_LOG_CRITICAL("Slot {}: the kernel did not drain the queue pair; the slot stays closed",
                          slot.name);
      return;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  reap(slot);

  // The connection ID is gone: free the bursts the application can no longer collect
  size_t dropped = slot.completions.size();
  size_t unsent =
      slot.closing_freed + slot.sq_inflight.size() + slot.rq_inflight.size() + slot.deferred.size();
  for (auto& in : slot.sq_inflight) {
    free_burst_internal(in.burst);
  }
  for (auto& in : slot.rq_inflight) {
    free_burst_internal(in.burst);
  }
  for (auto* burst : slot.deferred) {
    free_burst_internal(burst);
  }
  for (auto* burst : slot.completions) {
    free_burst_internal(burst);
  }
  BurstParams* burst = nullptr;
  while (slot.app_tx->dequeue(reinterpret_cast<void**>(&burst))) {
    free_burst_internal(burst);
    unsent++;
  }
  while (slot.app_rx->dequeue(reinterpret_cast<void**>(&burst))) {
    free_burst_internal(burst);
    dropped++;
  }
  if (dropped != 0 || unsent != 0) {
    DAQIRI_LOG_WARN(
        "Slot {}: dropped {} completed burst(s) the application did not collect and {} burst(s) "
        "that did not complete",
        slot.name, dropped, unsent);
  }
  slot.sq_inflight.clear();
  slot.rq_inflight.clear();
  slot.deferred.clear();
  slot.completions.clear();
  slot.sq_published = slot.rq_published = slot.sq_reaped = slot.rq_reaped = 0;
  slot.sq_pkt_head = slot.rq_pkt_head = 0;
  slot.sq_pkts_inflight = slot.rq_pkts_inflight = 0;
  slot.logged_error = 0;

  // The kernel no longer reads the rings: clear them for the next connection
  reset_qp(slot);
  std::memset(slot.sq_desc, 0, gr::DESC_RING * sizeof(gr::BurstDesc));
  std::memset(slot.rq_desc, 0, gr::DESC_RING * sizeof(gr::BurstDesc));
  slot.ctrl->sq_done = 0;
  slot.ctrl->rq_done = 0;
  slot.ctrl->error = 0;
  store_release(&slot.ctrl->state, static_cast<uint32_t>(gr::SLOT_IDLE));
  slot.closing = false;

  struct rdma_cm_id* id = nullptr;
  {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    id = slot.cm_id;
    slot.cm_id = nullptr;
    slot.claimed = false;
    slot.conn_id.store(0);
    slot.close_requested.store(false);
    slot.host_state.store(static_cast<uint32_t>(SlotHostState::FREE));
  }
  if (id != nullptr) {
    rdma_destroy_id(id);
  }
  DAQIRI_LOG_INFO("Slot {}: closed, ready for a new connection", slot.name);
}

void GpunetioRdmaEngine::worker(Slot* slot) {
  pin_thread(slot->cpu_core, "slot " + slot->name);
  if (!select_cuda_device(gpu_ordinal_, "running the worker of slot " + slot->name)) {
    return;
  }
  while (workers_running_.load(std::memory_order_relaxed)) {
    const auto state = static_cast<SlotHostState>(slot->host_state.load());
    if (state == SlotHostState::ACTIVE && slot->close_requested.load()) {
      close_slot(*slot);
      continue;
    }
    if (state != SlotHostState::ACTIVE) {
      std::this_thread::sleep_for(std::chrono::microseconds(200));
      continue;
    }

    bool busy = !slot->completions.empty();
    flush_completions(*slot);
    while (true) {
      BurstParams* burst = nullptr;
      const bool from_deferred = !slot->deferred.empty();
      if (from_deferred) {
        burst = slot->deferred.front();
      } else if (!slot->app_tx->dequeue(reinterpret_cast<void**>(&burst))) {
        break;
      }
      if (!publish(*slot, burst)) {
        if (!from_deferred) {
          slot->deferred.push_back(burst);
        }
        break;
      }
      if (from_deferred) {
        slot->deferred.pop_front();
      }
      busy = true;
    }
    const uint64_t before = slot->sq_reaped + slot->rq_reaped;
    reap(*slot);
    busy = busy || slot->sq_reaped + slot->rq_reaped != before;
    if (!busy) {
      cpu_relax();
    }
  }
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

void GpunetioRdmaEngine::shutdown() {
  if (!initialized_) {
    return;
  }
  DAQIRI_LOG_INFO("gpunetio_rdma engine shutting down");
  cm_running_.store(false);
  if (cm_thread_.joinable()) {
    cm_thread_.join();
  }

  // Tell the peers, then let the workers drain the connections
  {
    std::lock_guard<std::mutex> lock(slots_mutex_);
    for (auto& slot : slots_) {
      if (slot->host_state.load() == static_cast<uint32_t>(SlotHostState::ACTIVE)) {
        if (slot->cm_id != nullptr) {
          rdma_disconnect(slot->cm_id);
        }
        slot->close_requested.store(true);
      }
    }
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (auto& slot : slots_) {
    while (slot->host_state.load() == static_cast<uint32_t>(SlotHostState::ACTIVE) ||
           slot->host_state.load() == static_cast<uint32_t>(SlotHostState::CLOSING)) {
      if (std::chrono::steady_clock::now() - t0 > CLOSE_TIMEOUT) {
        DAQIRI_LOG_ERROR("Slot {} did not close in time", slot->name);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  workers_running_.store(false);
  for (auto& slot : slots_) {
    if (slot->thread.joinable()) {
      slot->thread.join();
    }
  }

  stop_kernel();
  destroy_resources();
  initialized_ = false;
}

// Releases everything initialize() created. Queue resources stay allocated if the kernel may still
// use them.
void GpunetioRdmaEngine::destroy_resources() {
  if (kernel_running_ && !stop_kernel()) {
    kernel_stuck_ = true;
  }
  for (auto& slot : slots_) {
    if (slot->cm_id != nullptr) {
      rdma_destroy_id(slot->cm_id);
      slot->cm_id = nullptr;
    }
    if (!kernel_stuck_) {
      destroy_queue_pair(*slot);
      free_pinned(slot->ctrl);
      free_pinned(slot->sq_desc);
      free_pinned(slot->rq_desc);
      free_pinned(slot->sq_pkts);
      free_pinned(slot->rq_pkts);
      slot->ctrl = nullptr;
      slot->sq_desc = slot->rq_desc = nullptr;
      slot->sq_pkts = slot->rq_pkts = nullptr;
    }
    if (slot->app_tx != nullptr) {
      daqiri::Ring::free(slot->app_tx);
      slot->app_tx = nullptr;
    }
    if (slot->app_rx != nullptr) {
      daqiri::Ring::free(slot->app_rx);
      slot->app_rx = nullptr;
    }
  }
  if (!kernel_stuck_) {
    slots_.clear();
    free_pinned(slot_args_);
    slot_args_ = nullptr;
    if (kernel_stream_ != nullptr) {
      cudaStreamDestroy(kernel_stream_);
      kernel_stream_ = nullptr;
    }
  }

  for (auto& listener : listeners_) {
    if (listener->id != nullptr) {
      rdma_destroy_id(listener->id);
    }
  }
  listeners_.clear();
  if (cm_channel_ != nullptr) {
    rdma_destroy_event_channel(cm_channel_);
    cm_channel_ = nullptr;
  }

  for (auto& device : devices_) {
    if (!kernel_stuck_) {
      if (device->dump_mr != nullptr) {
        ibv_dereg_mr(device->dump_mr);
      }
      if (device->dump_buf != nullptr) {
        cudaFree(device->dump_buf);
      }
    }
    for (auto& [name, mr] : device->mrs) {
      (void)name;
      ibv_dereg_mr(mr);
    }
    device->mrs.clear();
    if (!kernel_stuck_) {
      if (device->dev != nullptr) {
        doca_dev_close(device->dev);
      }
      if (device->vpd != nullptr) {
        doca_verbs_pd_destroy(device->vpd);
      }
      if (device->pd != nullptr) {
        ibv_dealloc_pd(device->pd);
      }
      if (device->vctx != nullptr) {
        doca_verbs_context_destroy(device->vctx);
      }
    }
  }
  if (!kernel_stuck_) {
    devices_.clear();
    if (gpu_ != nullptr) {
      doca_gpu_destroy(gpu_);
      gpu_ = nullptr;
    }
    if (cm_devices_ != nullptr) {
      rdma_free_devices(cm_devices_);
      cm_devices_ = nullptr;
    }
  }

  for (auto& [name, ring] : buf_pools_) {
    (void)name;
    daqiri::Ring::free(ring);
  }
  buf_pools_.clear();
  for (auto** pool : {&meta_pool_, &pkt_ptr_pool_, &pkt_len_pool_}) {
    if (*pool != nullptr) {
      daqiri::ObjectPool::free(*pool);
      *pool = nullptr;
    }
  }
}

void GpunetioRdmaEngine::print_stats() {
  DAQIRI_LOG_INFO("gpunetio_rdma engine stats");
  for (const auto& slot : slots_) {
    const auto state = static_cast<SlotHostState>(slot->host_state.load());
    const char* state_str = state == SlotHostState::ACTIVE       ? "active"
                            : state == SlotHostState::CONNECTING ? "connecting"
                            : state == SlotHostState::CLOSING    ? "closing"
                                                                 : "free";
    DAQIRI_LOG_INFO(
        "  slot {} ({} {}): {} connection(s), sent {} messages / {} bytes, received {} messages / "
        "{} bytes, {} failed burst(s)",
        slot->name, slot->server ? "server" : "client", state_str, slot->connections.load(),
        slot->send_msgs.load(), slot->send_bytes.load(), slot->recv_msgs.load(),
        slot->recv_bytes.load(), slot->failed_bursts.load());
  }
}

}  // namespace daqiri
