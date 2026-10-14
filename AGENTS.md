# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Build & run

```bash
# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DDAQIRI_BUILD_PYTHON=OFF -DDAQIRI_ENGINE="dpdk ibverbs"
cmake --build build -j
cmake --install build --prefix /opt/daqiri

# Container build (compiles patched DPDK from source)
BASE_TARGET=dpdk DAQIRI_ENGINE="dpdk ibverbs" scripts/build-container.sh
```

Container releases run locally. Run `PUSH=1 scripts/publish_container.sh` on
each supported architecture; after both per-architecture tags exist, run
`PUBLISH_MANIFEST=1 scripts/publish_container.sh` to publish and inspect the
canonical multi-architecture version tag.

CMake options (full table in `docs/tutorials/bare-metal-cmake-build.md`):
- `DAQIRI_ENGINE` — space-separated list of optional engines to compile. Valid values: `dpdk` (raw Ethernet), `ibverbs` (RDMA/RoCE), and the experimental `gpunetio_verbs` (NIC queues driven by CUDA kernels through DOCA Verbs and DOCA GPUNetIO, for raw Ethernet and RoCE; like `ibverbs` it builds two internal engines; needs DOCA 3.6+). Linux sockets (UDP/TCP) are always built in, so there is no `socket` value. Default is `"dpdk ibverbs"`.
- `DAQIRI_BUILD_PYTHON` — builds `pybind11` bindings from `python/`.
- `DAQIRI_BUILD_EXAMPLES` — builds the benchmark executables (default `ON`). The
  hardware-free `daqiri_config_validate` tool is always built and installed.
- `BUILD_TESTING` — builds and registers the hardware-free C++ tests under
  `tests/cpp/` with CTest (default `ON`). Set it to `OFF` to omit test targets.
- `DAQIRI_BUILD_APPLICATIONS` — builds the end-to-end example applications under `applications/` (default `OFF`; requires TensorRT, e.g. the `BASE_IMAGE=torch` container). Currently builds `applications/resnet50_inference/` (DAQIRI → TensorRT ResNet inference).
- `DAQIRI_ENABLE_OTEL_METRICS` — enables OpenTelemetry metrics instrumentation (default `OFF`).
- `DAQIRI_REORDER_GPU_PROFILE` — enable CUDA event timing in the DPDK reorder kernels (off by default).
- `DAQIRI_ENABLE_S3` — enable AWS SDK-backed asynchronous raw packet writes to S3 (off by default).
- `DAQIRI_PREFER_SYSTEM_YAML_CPP` — prefer system `yaml-cpp` over the vendored `third_party/yaml-cpp` submodule (off by default; keep off when a conda/miniforge env is on `PATH`).

Package versions use CalVer (`YYYY.MM.PATCH`) from the top-level `VERSION` file. CMake reads that value into `project(daqiri VERSION ...)`, generates `daqiri/version.h`, feeds pkg-config/CMake package metadata, and exposes the same value in Python. `DAQIRI_ABI_VERSION` is separate and currently `2`; do not tie ABI policy to the CalVer year.

CUDA architectures default to `80;90` (A100, H100), with `121` (GB10) added when configuring with CUDA Toolkit 13.0 or newer. Override `CMAKE_CUDA_ARCHITECTURES` when targeting other GPUs.

**Socket / ibverbs relationship**: the socket engine is always built and provides UDP/TCP directly; its RoCE path delegates to the `ibverbs` engine (internally the `rdma` engine — `src/engines/rdma/`, target `daqiri_rdma`, define `DAQIRI_ENGINE_RDMA`; `ibverbs` is only the user-facing name). `src/CMakeLists.txt` builds the ibverbs/rdma engine only when `ibverbs` is in `DAQIRI_ENGINE`, and the socket engine links it conditionally — so `socket` RoCE is available only when `ibverbs` was built, while plain UDP/TCP always works.

## Testing and benchmarks

Portable Python unit tests run without configuring or compiling DAQIRI:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --requirement tests/requirements.txt
.venv/bin/python -m pytest
```

The standard pre-PR check runs the portable tests, validates both retained and
generated configurations without initializing hardware, and checks the documentation:

```bash
scripts/check_pr.sh
```

The default pytest suite collects only `tests/portable/`. Build-backed C++ tests live under
`tests/cpp/` and run through CTest; Python-binding tests live under `tests/bindings/` and
require a container built with `DAQIRI_BUILD_PYTHON=ON`. Platform tests live under
`tests/platform/` and are selected by CI/CD jobs running on provisioned GPU/NIC systems;
they are never part of the default pytest collection. The project container already includes
the current test packages; use the container-specific dependency command in
`tests/README.md` when `tests/requirements.txt` changes.

Integration and performance verification is done via the benchmark executables in `examples/`, driven by YAML configs. Build outputs (`examples/CMakeLists.txt:59-71`):

| Executable | Source | Typical config |
|---|---|---|
| `daqiri_bench_raw_gpudirect` | `raw_gpudirect_bench.cpp` | Canonical examples: `daqiri_bench_raw_tx_rx.yaml`, `daqiri_bench_raw_tx_rx_4q.yaml`, `daqiri_bench_raw_sw_loopback.yaml`, `daqiri_bench_raw_hw_loopback_ibverbs.yaml`, `daqiri_bench_raw_rx_multi_q.yaml`, `daqiri_bench_raw_tx_rx_pacing.yaml` (per-queue `pacing_mbps`; DPDK engine only), `daqiri_bench_raw_tx_rx_gpunetio_verbs.yaml` (experimental gpunetio_verbs engine). Generate Spark, cross-host, multi-queue matrix, and VLAN/VXLAN/GRE/NVGRE variants with `scripts/gen_daqiri_config.py`; the Spark harnesses invoke it directly. |
| `daqiri_bench_raw_latency` | `raw_latency_bench.cpp` | `daqiri_bench_raw_latency_ibverbs.yaml` — caller-driven direct TX/RX, RX hardware timestamps, 64–8192-byte power-of-two latency sweep |
| `daqiri_example_dynamic_rx_flow` | `dynamic_rx_flow_example.cpp` | `daqiri_example_dynamic_rx_flow.yaml` — `flow_isolation: true` startup followed by runtime scalar queue steering, multi-queue RSS, and raw-engine decap/pop flow add/delete |
| `daqiri_example_dynamic_resource` | `dynamic_resource_example.cpp` | Any ibverbs config with at least one RX queue and one single-region TX queue (for example `daqiri_bench_raw_hw_loopback_ibverbs.yaml`) — initializes without RX queues, repeatedly adds/removes the first RX queue's steering flow, and exercises runtime MR and RX/TX queue add/delete |
| `daqiri_example_named_endpoints` | `named_endpoints_example.cpp` | `daqiri_example_named_endpoints_tx_rx.yaml` — raw ibverbs runtime named endpoints with inline Ethernet/IPv4/UDP headers and one gathered payload segment |
| `daqiri_bench_raw_hds` | `raw_hds_bench.cpp` | `daqiri_bench_raw_tx_rx_hds.yaml` |
| `daqiri_bench_raw_reorder_seq` | `raw_reorder_seq_bench.cpp` | `daqiri_bench_raw_tx_rx_reorder_seq_1024*.yaml`, `daqiri_bench_raw_rx_reorder_seq_*.yaml` |
| `daqiri_bench_raw_reorder_quantize` | `raw_reorder_quantize_bench.cpp` | `daqiri_bench_raw_tx_rx_reorder_quantize_seq_batch.yaml` |
| `daqiri_bench_rdma` | `rdma_bench.cpp` | Canonical example: `daqiri_bench_rdma_tx_rx.yaml`; `daqiri_bench_rdma_tx_rx_gpunetio_verbs.yaml` runs it on the experimental `gpunetio_verbs` RoCE engine. Generate Spark netns/cross-host client and server roles with `scripts/gen_daqiri_config.py socket-pair --transport roce`. |
| `daqiri_bench_socket` | `socket_bench.cpp` | Canonical examples: `daqiri_bench_socket_{udp,tcp}_tx_rx.yaml`. Generate Spark netns/cross-host client and server roles with `scripts/gen_daqiri_config.py socket-pair`. |
| `daqiri_pool_ring_bench` | `pool_ring_bench.cpp` | none — microbenchmark comparing `daqiri::Ring`/`daqiri::ObjectPool` vs DPDK `rte_ring`/`rte_mempool` (SPSC/MPMC, single/bulk, thread sweep). The `rte_*` comparison arm compiles only in a DPDK-enabled build; takes no YAML/CLI args |

The four `raw_*` benches share `raw_bench_common.{cpp,h}` and accept `--seconds N`. `daqiri_bench_rdma` and `daqiri_bench_socket` also take `--mode {server,client,both}`. `daqiri_bench_raw_gpudirect`, `daqiri_bench_raw_hds`, `daqiri_bench_rdma`, and `daqiri_bench_socket` additionally accept `--workload none|fft|gemm|gemm_fp16` — a reusable representative GPU workload (`examples/bench_workload.{h,cu}`, cuFFT/cuBLAS) run once per received reorder window on the **actual received payload**. Each backend first assembles the burst's payloads into one contiguous GPU buffer via `examples/bench_pipeline.{h,cu}` (`ReorderPipeline`): a sequence-number reorder kernel for the out-of-order transports (DPDK raw, UDP) and an arrival-order gather for the in-order ones (RoCE RC, TCP); sockets stage host→device first since their payloads land in pageable host memory. The reorder/gather kernels (`packet_reorder_copy_payload_by_sequence`, `packet_gather_copy_payload`) live in `src/kernels.cu`. `gemm` is FP32 `cublasSgemm`; `gemm_fp16` is the same-size mixed-precision FP16/tensor-core `cublasGemmEx` (inference-style); the contiguous buffer supplies the FFT input / GEMM A operand. `--workload-gemm-dim N` pins the square GEMM side length (default 1024), so the FLOP count per call (2·n³) is FIXED and the compute working set is exactly n·n·elem_size, read from the front of each received I/O unit (the unit must be at least that large). `--workload-fft-len N` pins the 1-D C2C transform length for `fft` (default 1024; independent of the GEMM dimension); the working set is fanned out across as many batched length-N transforms as fit. Used by `run_spark_bench.sh`'s `WORKLOAD` / `GEMM_DIM` / `FFT_LEN` env (all backends) to fill the CSV `post_process` / `post_process_gemm_dim` columns (issue #15).

```bash
./build/examples/daqiri_bench_raw_gpudirect ./build/examples/daqiri_bench_raw_tx_rx.yaml --seconds 10
./build/examples/daqiri_bench_socket        ./build/examples/daqiri_bench_socket_udp_tx_rx.yaml --seconds 10 --mode both
```

YAML files contain `<angle-bracket>` placeholders (PCIe addresses, CPU cores, MACs, IPs) that **must** be replaced for your system. `daqiri_bench_raw_sw_loopback.yaml` requires no NIC and is the fastest way to smoke-test a build. `daqiri_bench_raw_hw_loopback_ibverbs.yaml` uses one enumerated mlx5 port without a cable to exercise real TX, RX, and flow steering in hardware.

Configs named `raw_rx_*` are RX-only — they initialize the RX path and wait for external traffic, so a standalone run can exit cleanly with `0` packets. Use the `tx_rx` configs for closed-loop smoke tests. NIC flow programming (RX flows, dynamic RX flows, `tx_eth_src`) requires a hardware loopback or cross-host wire test — see `docs/benchmarks/raw_benchmarking.md` (Flow programming smoke test); `daqiri_bench_raw_sw_loopback.yaml` only smoke-tests the build/runtime path.

When determining throughput for a benchmark use the `mlnx_perf` utility in the background to view
transmit and receive rates. Using application run time with packet counts is usually not accurate
enough due to startup inconsistencies. Hardware-loopback performance testing is different from a
cabled test: use `mlnx_perf -i <netdev> -t 1` and report `vport_loopback_bytes` as the aggregate
single-port loopback rate. Do not add TX and RX rates for the same returned traffic. The
`*_bytes_phy`/`*_packets_phy` counters should remain flat because no packet crosses the SerDes.
Discard partial startup/shutdown samples and report stable one-second samples from a run of at
least 10 seconds. Label the result as **hardware-loopback throughput**, not wire throughput: it
excludes the cable and peer host and, with device memory, measures the local NIC's internal return
path plus one GPUDirect read from GPU memory and one GPUDirect write back to GPU memory per packet.
It can therefore differ from, or exceed, the port's physical line rate. Use DAQIRI per-queue stats
to verify steering and drops; the vport counter is aggregate rather than per queue.

### Example applications (`applications/`, opt-in)

Beyond the benchmarks, end-to-end **example applications** live under `applications/` and build only with `-DDAQIRI_BUILD_APPLICATIONS=ON` (off by default; they pull in heavier optional dependencies). They are downstream `daqiri::daqiri` consumers — they reuse the bench scaffolding (`examples/raw_bench_common.*`, `bench_pipeline.cu`) but are not part of the benchmark table above.

| Executable | Source | Typical config |
|---|---|---|
| `daqiri_resnet50_inference` | `applications/resnet50_inference/` | `configs/resnet50_{tx,rx}_spark_xhost.yaml` (Spark-to-Spark: TX on stacked-01, RX+inference on stacked-02). Config-based GPU reorder + int8→fp16 quantize → TensorRT FP16 tensor-core inference, RX/inference decoupled via SPSC ring. Also `resnet50_sw_loopback.yaml` (no NIC), `resnet50_wire_loopback.yaml` / `resnet50_bench_spark.yaml` for optional single-host smoke. Requires TensorRT (`BASE_IMAGE=torch`). See `docs/tutorials/daqiri-resnet-inference.md`. |

## Formatting

`clang-format` is required for contributions (CONTRIBUTING.md):

```bash
git-clang-format --style file              # format staged changes
clang-format -style=file -i -fallback-style=none <files>
```

## Architecture

**Single C++/CUDA shared library** (`libdaqiri.so`) exposing a C++ API through `#include <daqiri/daqiri.h>`. The public surface is intentionally flat free-function helpers (`get_rx_burst`, `get_packet_ptr`, `set_udp_header`, `socket_setsockopt`, runtime flow/resource operations, …) that all operate on opaque DAQIRI-owned buffers, operation IDs, or connection IDs. Applications never touch engine types directly.

`include/daqiri/daqiri.h` also includes the generated `daqiri/version.h`, exposing `DAQIRI_VERSION`, CalVer component macros, `DAQIRI_ABI_VERSION`, and inline helpers such as `daqiri::version_string()` and `daqiri::abi_version()`.

### Engine abstraction
`src/engine.h` defines `daqiri::Engine` — an (almost) ABC with ~50 virtual methods covering init, RX/TX burst dequeue/enqueue, header-fill helpers, buffer free, socket connection helpers, runtime TCP/UDP `setsockopt` passthrough, and RDMA connection setup. Engines live in `src/engines/<name>/` (`dpdk/`, `rdma/`, `socket/`, `ibverbs/`, `gpunetio_verbs/`, `gpunetio_rdma/`). `DAQIRI_ENGINE` selects the optional `dpdk`, `ibverbs`, and `gpunetio_verbs` engines at CMake configure time; the `socket` engine is always built. The user-facing value `ibverbs` builds **two** internal engines that both use libibverbs: `rdma` (`src/engines/rdma/`, `DAQIRI_ENGINE_RDMA`, RoCE/InfiniBand for socket `roce://`) and `ibverbs` (`src/engines/ibverbs/`, `DAQIRI_ENGINE_IBVERBS`, the pure-DevX MPRQ raw-Ethernet engine). The user-facing value `gpunetio_verbs` likewise builds two internal engines on DOCA Verbs and DOCA GPUNetIO: `gpunetio_rdma` (`src/engines/gpunetio_rdma/`, `DAQIRI_ENGINE_GPUNETIO_RDMA`, RoCE for socket `roce://`) and `gpunetio_verbs` (`src/engines/gpunetio_verbs/`, `DAQIRI_ENGINE_GPUNETIO_VERBS`, raw Ethernet). Each engine produces its own static library (`daqiri_dpdk`, `daqiri_rdma`, `daqiri_socket`, `daqiri_ibverbs`, `daqiri_gpunetio_verbs`, `daqiri_gpunetio_rdma`) linked into `daqiri_common`, and each adds a `DAQIRI_ENGINE_<NAME>=1` compile definition.

`EngineType` (`include/daqiri/types.h`) is resolved from `(stream_type, engine)`: `raw` defaults to `EngineType::IBVERBS` when that engine is built (falling back to `EngineType::DPDK` in DPDK-only builds); `raw` + `engine: "dpdk"` explicitly selects `EngineType::DPDK`; `raw` + `engine: "gpunetio_verbs"` selects `EngineType::GPUNETIO_VERBS` and `socket` + `engine: "gpunetio_verbs"` selects `EngineType::GPUNETIO_RDMA` (never a default); `socket` + a `roce://` endpoint (or `engine: "ibverbs"`) selects `EngineType::RDMA`. The stream-aware `config_engine_from_string(str, stream_type)` overload encodes the `ibverbs`→`{IBVERBS for raw, RDMA for socket}` and `gpunetio_verbs`→`{GPUNETIO_VERBS for raw, GPUNETIO_RDMA for socket}` splits; `rdma` and `gpunetio_rdma` are rejected in configs. `EngineFactory` (in `engine.h`) is a singleton that instantiates the active engine. `daqiri_init(...)` resolves which engine to use from the `NetworkConfig`, runs the shared hardware-independent semantic validation, and only then delegates initialization through the `Engine` vtable. The standalone `daqiri_config_validate` tool calls the same parser and shared checks without creating an engine. There is only ever **one** active `Engine` per process.

The always-built socket engine implements Linux UDP/TCP streams directly. Applications that need kernel socket tuning call `socket_setsockopt(conn_id, level, optname, optval, optlen)` after resolving a TCP/UDP connection ID; DAQIRI passes the numeric Linux constants through without maintaining a symbolic option map. TCP RX queues bound their internal burst backlog by the smallest `num_bufs` among the queue's referenced memory regions. Receive threads wait for readable data without reserving shared queue capacity, then reserve a slot before consuming; a full queue stops socket consumption so TCP backpressure reaches the sender, while an idle peer cannot monopolize capacity shared with active peers. UDP retains its existing receive behavior. `socket_setsockopt` is not supported for `roce://` connections, which delegate to the RDMA/ibverbs path.

The opt-in `dpdk` raw engine (`src/engines/dpdk/`, `DpdkEngine`) programs RX steering, send-to-kernel fallbacks (`flow_isolation: true`), and `tx_eth_src` TX offloads via DPDK RTE Flow during `daqiri_init()`. Standard UDP/IP (group 3), flex-item (group 1) and eCPRI-over-Ethernet (group 2, EtherType `0xAEFE`, via `RTE_FLOW_ITEM_TYPE_ECPRI` matching message type and pc_id/rtc_id) RX flows use separate flow groups; `validate_config()` rejects mixing these flow classes per interface, duplicate `flex_item_id` values, and unknown queue targets / `flex_item_id` values before NIC programming. Queue actions with two or more IDs use mlx5 Toeplitz IPv4/UDP RSS. The mlx5 PMD honors the native eCPRI flow item only under **firmware steering** (`dv_flow_en=1`); under HW steering (`dv_flow_en=2`, the default) the rule installs but silently never matches on ConnectX-class NICs. So `initialize()` auto-switches any interface carrying eCPRI RX flows to `dv_flow_en=1` (with a `WARN`), which means the async/template dynamic-RX-flow path is unavailable on that port. Flex-item parser handles are created per `(port, flex_item_id)` (scoped per interface). All programmed `rte_flow` rules and flex-item handles are tracked and destroyed in order on shutdown, init failure, and engine teardown (programmed flows → flex items → group-0 ETH jump rules). See `docs/benchmarks/raw_benchmarking.md` (Flow programming smoke test) for manual verification steps. It also reports 802.3x pause state at init and per-run pause counter deltas from `src/net_pause.{h,cpp}` when a kernel netdev is available.

The `ibverbs` raw engine (`src/engines/ibverbs/`, `IbverbsEngine`) drives a Mellanox/mlx5 Multi-Packet (striding) Receive Queue via **DevX** (`mlx5dv_devx_obj_create` against vendored PRM structs in `mlx5_prm_min.h`): a DevX CQ + striding RQ + TIR + `mlx5dv_dr` flow steering, with manual WQE/doorbell management and worker-driven cyclic refill. RX packets DMA strided into one pre-posted MR (host or GPU via `ibv_reg_dmabuf_mr`); a queue with >1 memory region instead uses a non-striding DevX *regular* RQ with multi-segment scatter WQEs for **physical** header-data split (header → CPU MR, payload → GPU MR). TX builds mlx5 send WQEs directly on a raw-packet QP's SQ (via `mlx5dv_init_obj`, bypassing `ibv_post_send`) from a slab of registered slots tracked by cyclic index counters, with NIC checksum offload and a `tx_eth_src` offload. Runtime named endpoints use a preallocated NUMA-local slot table with process-lifetime generation-tagged IDs and reader quiescence for deletion; submission snapshots the cached Ethernet/IPv4/UDP header without a name-map lookup, inlines it in the mlx5 SEND WQE, and gathers the payload from the caller-selected TX queue. `loopback: "hw"` enables single-port unicast self-loopback through a retained mlx5dv activation QP and opts both direct and RSS TIRs into receiving the internally returned packets. It uses the libdpdk-free `daqiri::Ring`/`daqiri::ObjectPool` for the worker→app burst handoff (like the rdma engine — neither links DPDK) and drives the NIC through libibverbs/mlx5dv directly. Feature set: RX (MPRQ), TX, GPUDirect, physical/logical HDS, multi-queue 5-tuple flow steering with per-packet flow IDs, flex-item arbitrary-offset, IPv4-total-length and eCPRI-over-Ethernet (EtherType `0xAEFE`, message type + pc_id/rtc_id) flow matching (mlx5 flex parser / `misc_parameters_4`), per-packet RX hardware timestamps, per-queue hardware packet pacing through the mlx5 QP rate table, accurate TX send scheduling (wait-on-time WAIT WQE), GPU/CPU software reorder and quantize, and ConnectX-7+ first-DMA hardware reorder. Hardware reorder is explicit opt-in (`reorder_engine: "hw"`, `cyclic_sequence: true`): the mlx5 flex parser steers finite-ring sequence values to private fixed-slot RQs, the host poller aggregates CQEs, and destinations may be CPU or GPU memory. It does not use DPA. Exact 32-bit parser-sample matching means wide monotonic sequence fields are unsupported; use software reorder for those streams. Each destination exposes one receive credit, and a direct-placed batch is rearmed only when the application frees its burst, preventing DMA into caller-owned output. Packet pacing and accurate send are independent: `pacing_mbps` configures the QP's average rate, while `set_packet_tx_time()` emits WAIT WQEs for absolute per-packet times. Because it uses the kernel netdev directly, `ensure_port_mtus` raises the netdev MTU at init to cover the configured frame size in either direction — RX (post-decap) and TX egress (post-encap) — sizing each direction with its own transform wire overhead (jumbo frames silently drop otherwise). It also uses `src/net_pause.{h,cpp}` to warn when pause is enabled and to report per-run pause frame deltas from `print_stats()`. Queues sharing a `cpu_core` are serviced round-robin by one poller thread.

The experimental `gpunetio_verbs` raw engine (`src/engines/gpunetio_verbs/`, `GpunetioVerbsEngine`, CUDA kernels in `daqiri_gpunetio_verbs_kernels.cu`) runs each queue as a DOCA Verbs Ethernet queue (`doca_verbs_eth_rq` / `doca_verbs_eth_sq`, DOCA 3.6 or newer, pkg-config `doca-gpunetio`, `doca-verbs`, `doca-common`, plus libibverbs and libmlx5) whose work queue, doorbell record and completion queue (overrun mode) live in GPU memory, and drives it from the GPU with DOCA GPUNetIO. It uses no DOCA Flow and no DOCA Ethernet. It is selected only by an explicit `engine: "gpunetio_verbs"`. Each interface (PCIe address, IB device name or netdev name, matched with DOCA devinfo) opens its mlx5 port with `mlx5dv_open_device` (DevX) and an `ibv_pd`, imported into DOCA Verbs with the bridge (`doca_verbs_bridge_verbs_context_import`/`_pd_import`, `doca_rdma_bridge_open_dev_from_pd`). Memory regions are registered with `ibv_reg_dmabuf_mr` (`device`) or `ibv_reg_mr` (`host_pinned`). Each queue is exported with `doca_gpu_verbs_bridge_export_qp` (`GPU_SM_DB`), which takes a send and a receive side: an RX queue's unused send side points at its own ring and the unused send counter of its doorbell record (with a UAR of its own for the doorbell mapping), a TX queue's unused receive side at the unused receive counter. The kernels use the DOCA GPUNetIO device functions for doorbells (`doca_gpu_dev_verbs_submit`) and WQE stores, and poll the completion queues with their own owner-bit check (plus the work-request counter before Hopper, like DOCA). RX: receive work request `i` points at slot `i` of the queue's exclusive memory region (64-byte-multiple slots, a power-of-two ring of at most 32768, rounded up in regions DAQIRI allocates and the first slots of regions the application binds), written once from the host; one resident RX kernel, one block per queue, polls completions block-parallel (a contiguous prefix of positions some thread saw complete), writes byte counts and steering tags (flow IDs, low 24 bits of `sop_drop_qpn`) to pinned host memory, publishes bursts that never wrap the ring, and reposts work requests (doorbell record = ring size + released packets) only once the RX worker reports the application freed them, oldest first. TX: cyclic slots of the queue's exclusive region (as the ibverbs engine); one resident TX kernel, one block per queue, takes bursts from a descriptor ring in pinned host memory and posts one single-WQEBB SEND per packet (control, Ethernet segment with L3/L4 checksum offload and no inline headers, data segment), a completion closing each chunk of at most 1/8 of the send queue. Steering uses rdma-core directly: a DevX transport domain, a direct TIR per queue (or an RQT and a Toeplitz IPv4/UDP TIR for several queue IDs) on the DOCA RQ numbers, and `mlx5dv_dr` rules in the NIC RX root table, with tag actions for the flow IDs; unmatched packets go to the kernel with `flow_isolation: true`, otherwise to the first RX queue. Root-table rules take matcher priorities 0-15 only (mlx5_ib), so an interface takes at most 16 rules (its flows plus the catch-all), checked in validation. Only one interface per port may have RX queues. One worker thread per `cpu_core` serves the RX and TX queues pinned to it in turn. On GPUs that don't order GPUDirect RDMA writes for their own kernels, the RX workers call `cuFlushGPUDirectRDMAWrites()` before handing over bursts in `device` memory. Shared-config limits (one region per queue, not shared, `device`/`host_pinned`, one GPU, `batch_size` at most the region's `num_bufs`, IPv4/UDP or Ethernet matches without `ipv4_len`, 24-bit flow IDs, 16 steering rules per interface, no offloads/pacing/accurate send/timestamps/dynamic flows/flex items/reorder/software loopback) live in `validate_network_config()`. Raw queues and steering need root. The resident kernels run until shutdown, so the usual `cudaFree()`/`cudaDeviceSynchronize()` and CUDA lazy-loading caveats apply (the engine preloads its kernels and warns unless `CUDA_MODULE_LOADING=EAGER`).

The experimental `gpunetio_rdma` engine (`src/engines/gpunetio_rdma/`, `GpunetioRdmaEngine`, CUDA kernel in `daqiri_gpunetio_rdma_kernels.cu`; user-facing name `gpunetio_verbs` on socket streams) keeps the API and the connection model of the `rdma` engine (`rdma_connect_to_server`, `rdma_get_server_conn_id`, SEND/RECEIVE bursts returned by `get_rx_burst(conn_id)`) and drives the NIC queues from the GPU with DOCA Verbs and DOCA GPUNetIO (plus librdmacm). It is selected only by an explicit `engine: "gpunetio_verbs"` on a `socket` stream with `roce://` endpoints. Each queue slot (one per RX queue on a server interface, one per TX queue on a client interface) owns a DOCA Verbs RC queue pair created at init on the interface's RDMA device (`rdma_cm` binds the address to find it; the `ibv_context` and `ibv_pd` are imported with the DOCA Verbs bridge), with its 1024-entry work queues, completion queues (overrun mode) and doorbell records in GPU memory, exported to the GPU with `doca_gpu_verbs_export_qp` (`GPU_SM_DB` required). `rdma_cm` connects the peers without owning the queue pair: the engine moves it through INIT/RTR/RTS with the attributes of `rdma_init_qp_attr()`, converted by `doca_verbs_bridge_qp_attr_set()` (the RD atomic limits are set separately, the bridge drops them), and passes its QPN in `rdma_conn_param` (`rdma_accept` on the server, `rdma_connect` + `RDMA_CM_EVENT_CONNECT_RESPONSE` + `rdma_establish` on the client), so it interoperates with `ibverbs` RoCE peers. One resident kernel on the GPU of the `device` memory regions (or CUDA device 0) has one block per slot: the slot's CPU worker publishes SEND and RECEIVE bursts on descriptor rings in pinned host memory (memory keys big endian, as the DOCA WQE functions take them), thread 0 reserves the WQEs, the block writes them (one signaled WQE per message), and thread 0 rings the doorbell after a release fence, then polls both completion queues without blocking, writes received byte counts back, and reports bursts in order through counters. Error completions are consumed by hand (the DOCA poll functions advance `cqe_ci` only on success). On pre-Hopper GPUs (`need_mcst`) a receive burst into GPU memory is followed by a DUMP WQE, and the host keeps send-queue room for those. Connection IDs encode `(generation << 32) | (slot + 1)`. On a disconnect the worker blocks the application calls on the slot, moves the queue pair to ERR, lets the block drain every posted WQE and reset the GPU queue pair (indexes to 0, `cqe_rsvd` advanced by the posted WQEs, doorbell records cleared), then resets the queue pair to RST, so the slot takes the next connection without any CUDA call (freeing GPU memory would block on the resident kernel). Shared-config checks (RC only, one GPU for the `device` regions, queue slots present) live in `validate_network_config()`. RC queue pairs need no root. The kernel runs until shutdown, so the same `cudaFree()`/`cudaDeviceSynchronize()` and CUDA lazy-loading caveats as any resident kernel apply (the engine preloads its kernel and warns unless `CUDA_MODULE_LOADING=EAGER`).

Raw ibverbs pools DAQIRI-owned startup `MemoryKind::HUGE` regions with the same NUMA affinity in one hugetlb arena while retaining separate logical regions and registrations. Runtime regions are allocated independently. `MemoryKind::HUGE` is strict across engines: allocation failure aborts the operation rather than silently falling back to regular or transparent-hugepage memory.

### Zero-copy / BurstParams
All packet data flows through `BurstParams`, a batch of packets. Only pointers are passed between NIC, DAQIRI internals, and the application — the caller reads directly from the buffers the NIC DMA'd into. **The caller must explicitly free bursts**; a missed free drains the pool and produces `NO_FREE_BURST_BUFFERS` / `NO_FREE_PACKET_BUFFERS` errors and NIC drops. See `docs/concepts.md` (Zero-Copy Ownership) and `docs/api-reference/cpp.md` (free-function call patterns).

### Segments & HDS
A single packet can span multiple **segments** (contiguous memory regions), each in CPU or GPU memory. The header-data split (HDS) mode puts headers in segment 0 (CPU) and payload in segment 1 (GPU), enabling GPUDirect zero-copy payload paths. Batched-GPU and CPU-only modes use a single segment.

### DPDK is required only by the `dpdk` engine
DPDK is **not** a dependency of `daqiri_common` or of the `rdma`/`ibverbs`/`socket` engines. Those use the libdpdk-free `daqiri::Ring` (`src/daqiri_ring.h`) and `daqiri::ObjectPool` (`src/daqiri_pool.h`) — header-only replacements for the generic `rte_ring` (a bounded MPMC/SPSC pointer ring, the DPDK C11 4-cursor algorithm) and `rte_mempool` (a fixed-size slab + free-ring) usage. `src/CMakeLists.txt` parses `DAQIRI_ENGINE` first, sets `DAQIRI_BUILD_DPDK`, and only then runs `pkg_check_modules(DPDK REQUIRED libdpdk)` and links it into `daqiri_common`. The only `rte_*` code left in `daqiri_common` lives in `dpdk_log.cpp` and `src/engine_dpdk.cpp` (the DPDK-only `Engine` base-class methods: mbuf/extmem registration, mbuf packet pools, EAL/hugepage preflight + cleanup), both compiled in **only** when `dpdk ∈ DAQIRI_ENGINE`. `net_pause.cpp` is compiled into `daqiri_common` unconditionally and uses only Linux ethtool ioctls, so it adds no DPDK or libibverbs dependency. The `rdma`/`ibverbs` engines no longer call `rte_eal_init` at all (they only needed EAL to back the rings/pools); the `dpdk` engine still does. `MemoryKind::HUGE` allocation goes through the virtual `Engine::alloc_huge` hook — `mmap(MAP_HUGETLB)` in the base, overridden by `DpdkEngine` to use `rte_malloc_socket`. A build with `DAQIRI_ENGINE="ibverbs"` (or `""`) links no `librte_*`. **libnuma is an optional dependency** (`DAQIRI_HAVE_NUMA`, auto-detected): when present, `daqiri::Ring`/`daqiri::ObjectPool` and `alloc_huge` pin their memory to the NUMA node DPDK's socket-aware allocators used (rings/pools → the master core's node, mirroring `rte_socket_id()`; HUGE MRs → `mr.affinity_`, mirroring `rte_malloc_socket`); without it they fall back to first-touch placement. `python/tune_system.py --check numa` warns when libnuma is absent on a multi-socket host. The container build still uses patched DPDK from `dpdk_patches/` (`dmabuf.patch`, `dpdk.nvidia.patch`) for the `dpdk` engine — the `dmabuf` patch removes the peermem kernel-module requirement for GPUDirect.

The base and DPDK `MemoryKind::HUGE` allocators are both strict: neither substitutes regular memory when hugetlb/EAL hugepage allocation fails.

### Reorder & quantize kernels
`src/kernels.cu` hosts the CUDA reorder paths used by the `raw_reorder_*` benches. Compile with `-DDAQIRI_REORDER_GPU_PROFILE=ON` to instrument them with CUDA event timing.

### Third-party dependencies
Vendored under `third_party/` as submodules (`.gitmodules`): `yaml-cpp` (config parser) and `spdlog` (logging). CMake prefers these over system copies. Missing them is a fatal error.

### Current limitations
- TX header fill currently supports UDP only (see README).
- Raw Ethernet RX flow scalar `action.id` or every entry in `action.ids` (including the final `actions:` queue action) must match an `rx.queues` ID, and flex-item flows must reference a valid `flex_item_id` on the same interface; `daqiri_init()` aborts if RX flow rules, RSS destinations, send-to-kernel fallbacks (`flow_isolation: true`), transform flow actions, or `tx_eth_src` offload rules cannot be programmed on the NIC.
- Raw Ethernet tunnel/VLAN transform flows are hardware-only on the DPDK and ibverbs raw engines. TX flows may contain only push/encap transform actions and RX transform flows must use pop/decap actions ending in a queue; socket/RDMA engines reject these actions instead of adding a software fallback.
- Raw Ethernet RX flow steering: a single interface cannot mix standard (UDP/IP) and
  flex-item flows, and flex-item flows cannot combine with tunnel/VLAN transform actions; `DpdkEngine::validate_config()` rejects mixed configs at init.

## Documentation

The web docs live in `docs/` and are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). The site config is `mkdocs.yml`.

**Structure:**
- `docs/index.md` — landing page orchestrator (includes `docs/landing/*.html` snippets via pymdownx snippets)
- `docs/landing/` — landing section HTML fragments (hero, features, quick start, examples, tutorials, news, CTA, footer, overlay)
- `overrides/home.html` — Material theme override for the landing layout
- `docs/getting-started.md` — system requirements, build instructions, and first benchmark smoke-test guidance. Only add information to Getting Started when it directly affects requirements, library build steps, or benchmark smoke-test instructions.
- `docs/concepts.md` — terminology glossary (stream types and endpoint URI schemes, GPUDirect, packet/burst/segment, flow/queue, memory region, zero-copy ownership, RX reorder). Meant to be opened in parallel with the rest of the docs.
- `docs/api-reference/index.md` — API guide (6-step application lifecycle, configuration-first model)
- `docs/api-reference/configuration.md`, `docs/api-reference/cpp.md`, `docs/api-reference/python.md` — YAML reference, C++ API, and Python bindings docs
- `docs/tutorials/` — tutorial walkthroughs (system config, config-file walkthrough, Holoscan integration, ResNet inference)
- `docs/benchmarks/` — benchmark guide pages, surfaced as a top-level "Benchmarking" nav section in `mkdocs.yml` and the landing page (`docs/index.md`):
  - `docs/benchmarks/index.md` — overview and engine-selection decision tree
  - `docs/benchmarks/socket_benchmarking.md` — "Socket and RDMA Benchmarking" (TCP/UDP and RoCE/RDMA)
  - `docs/benchmarks/raw_benchmarking.md` — "Raw Ethernet Benchmarking" (DPDK `raw_*` benches)
  - `docs/benchmarks/performance-dgx-spark.md` — per-platform performance report for DGX Spark stream/protocol combinations (the long internal report lives outside the repo in `projects/daqiri-notes/`)
- `docs/stylesheets/extra.css` — custom theme overrides

**User-facing vocabulary:** the YAML format uses `stream_type` (`raw`, `socket`, future `pcie`); for socket streams the transport is encoded in the endpoint URI scheme (`udp://`, `tcp://`, `roce://`) in `socket_config.local_addr`/`remote_addr`, **not** a separate `protocol` field. (`SocketProtocol` still exists internally, derived from the scheme.) **"Engine"** is the standard term for the specific library backing an implementation; it replaced the former "manager" and "backend" terms and is now used consistently across code (`src/engines/<name>/`, the `Engine` ABC, CMake `DAQIRI_ENGINE`), the API reference, tutorials, the landing page, and concept pages. The mapping: `stream_type: "raw"` is implemented by the `dpdk` engine (or by the experimental `gpunetio_verbs` engine with `engine: "gpunetio_verbs"`); `stream_type: "socket"` with `udp://`/`tcp://` endpoints by the always-built `socket` engine; `stream_type: "socket"` with `roce://` endpoints by the `ibverbs` engine (or by the experimental `gpunetio_verbs` engine with `engine: "gpunetio_verbs"`).

**Keeping docs in sync with code:** before committing changes, scan for the recurring drift hotspots:
- **Stream-type list** (`src/engines/*/`) — README Engines table, `docs/getting-started.md`, `docs/concepts.md` (Stream Types section + Support and testing admonition), `docs/api-reference/configuration.md`
- **CMake options / `DAQIRI_ENGINE` default** (`src/CMakeLists.txt`) — README Quick Start, `docs/getting-started.md`, this file's Build & run section
- **Benchmark binary or YAML names** (`examples/`) — the benchmark table above, `docs/benchmarks/raw_benchmarking.md`, the "Choosing an example config" decision tree in `docs/tutorials/configuration-walkthrough.md` (every YAML must have a leaf; CI's `scripts/check_doc_refs.py` enforces coverage), and per-platform performance docs (`docs/benchmarks/performance-*.md`)
- **Public API include** (`#include <daqiri/daqiri.h>`; source files under `include/daqiri/`) — `docs/api-reference/index.md`, `docs/api-reference/cpp.md`, `docs/api-reference/python.md`; if the change adds or renames a user-facing concept, also `docs/concepts.md`
- **Python bindings** (`python/daqiri_common_pybind.cpp`) — `docs/api-reference/python.md` (function reference tables, enums/classes tables, GIL Behavior section)
- **Bench CLI flags or output format** (`examples/raw_bench_common.{h,cpp}`, `*_bench.cpp`) — per-platform performance docs' Methodology section, `examples/run_spark_bench.sh` parsing logic
- **Doc reorganization** (any rename in `docs/`) — `docs/index.md` landing page, `mkdocs.yml` nav, README Documentation table

The full mapping with rationale lives in the docs-sync agent rule. Internal-link, anchor, and nav drift is enforced by CI (`.github/workflows/docs.yml`); content drift (stale binary names, defaults) is still a manual check at commit time.

**Deployment:** `.github/workflows/docs.yml` runs `mkdocs gh-deploy --force` on pushes to `main`, publishing to the `gh-pages` branch. GitHub Pages serves from `gh-pages`.

## Contribution rules

From `CONTRIBUTING.md`:
- **DCO sign-off required** — every commit must have `Signed-off-by:` (use `git commit -s`). Unsigned commits will be rejected.
- Commit titles in imperative mood, prefixed with the GitHub issue number: `#<Issue Number> - <Title>`.
- An issue must exist and be approved before coding.
- Prefer toggling features via new CMake options (with backward-compatible defaults) rather than wrapping entire files in `#if` guards. Use `#if` only for minor in-file changes.
- Keep PRs narrowly scoped — one concern per PR, dependencies noted in the description.
- Run `scripts/check_pr.sh` before opening or updating every PR. If the PR changes
  anything under `docs/images/packet_diagrams/`, run `scripts/check_pr.sh --diagrams`.
  If it changes the Docker base stage, add `--docker-base`; combine both flags when
  both areas change.
- When opening a PR that touches `src/`, `examples/`, or `mkdocs.yml`, scan the doc-sync agent rule and update affected docs in the same PR.

## Compiling and Running

Compiling should always be done inside of the container built from the project's Dockerfile. The container should be started in privileged mode with all GPUs passed through. Hugepages mounted on the host should be passed through into the container via a volume mount. When compiling the container should be started with the current user. When running the  
container should run as root.
