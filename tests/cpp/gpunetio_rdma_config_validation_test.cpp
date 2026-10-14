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

// Checks, without a GPU or NIC, how the configuration parser and the shared validation handle
// engine: gpunetio_verbs on socket streams, which selects the gpunetio_rdma engine (RoCE), as
// engine: ibverbs selects the rdma engine. Nothing here creates the engine or opens a device.

#include <cstdio>
#include <string>
#include <vector>

#include <daqiri/daqiri.h>

#include "src/engine.h"

namespace {

const char* const kBaseConfig = R"yaml(%YAML 1.2
---
daqiri:
  cfg:
    version: 1
    stream_type: "socket"
    engine: "gpunetio_verbs"
    master_core: 0
    log_level: "info"
    memory_regions:
    - name: "DATA_CLIENT"
      kind: "device"
      affinity: 0
      num_bufs: 16
      buf_size: 1048576
    - name: "DATA_SERVER"
      kind: "device"
      affinity: 0
      num_bufs: 16
      buf_size: 1048576
    interfaces:
    - name: "client"
      address: "10.0.0.1"
      socket_config:
        mode: client
        local_addr: "roce://10.0.0.1"
      roce_config:
        transport_mode: RC
      tx:
        queues:
        - name: "client_tx"
          id: 0
          cpu_core: 1
          batch_size: 1
    - name: "server"
      address: "10.0.0.2"
      socket_config:
        mode: server
        local_addr: "roce://10.0.0.2:4096"
      roce_config:
        transport_mode: RC
      rx:
        queues:
        - name: "server_rx"
          id: 0
          cpu_core: 2
          batch_size: 1
)yaml";

struct Case {
  const char* name;
  const char* from;
  const char* to;
};

// Configurations the parser accepts and the shared validation rejects
const std::vector<Case> kUnsupportedCases = {
    {"UC transport", "      roce_config:\n        transport_mode: RC\n      tx:",
     "      roce_config:\n        transport_mode: UC\n      tx:"},
    // The kernel and the queue rings run on one GPU
    {"device memory on two GPUs",
     "    - name: \"DATA_SERVER\"\n      kind: \"device\"\n      affinity: 0",
     "    - name: \"DATA_SERVER\"\n      kind: \"device\"\n      affinity: 1"},
    // A server interface has one connection slot per RX queue
    {"server without RX queues",
     "      rx:\n        queues:\n        - name: \"server_rx\"\n          id: 0\n          "
     "cpu_core: 2",
     "      tx:\n        queues:\n        - name: \"server_rx\"\n          id: 0\n          "
     "cpu_core: 2"},
};

// Configurations the parser rejects
const std::vector<Case> kInvalidCases = {
    {"TCP endpoint", "        local_addr: \"roce://10.0.0.1\"",
     "        local_addr: \"tcp://10.0.0.1:5000\""},
    // As with "rdma" for "ibverbs", the internal name is not valid in a config
    {"internal engine name", "    engine: \"gpunetio_verbs\"", "    engine: \"gpunetio_rdma\""},
};

// Replaces the first occurrence of from, which must exist
bool patch(std::string* yaml, const std::string& from, const std::string& to) {
  const size_t pos = yaml->find(from);
  if (pos == std::string::npos) {
    return false;
  }
  yaml->replace(pos, from.size(), to);
  return true;
}

bool parse(const std::string& yaml, daqiri::NetworkConfig* config) {
  return daqiri::parse_network_config_from_yaml_string(yaml, *config) == daqiri::Status::SUCCESS;
}

}  // namespace

int main() {
  int failures = 0;
  const auto expect = [&failures](bool condition, const std::string& what) {
    if (!condition) {
      std::fprintf(stderr, "FAILED: %s\n", what.c_str());
      failures++;
    }
  };

  daqiri::NetworkConfig base;
  expect(parse(kBaseConfig, &base), "base config parses");
  expect(base.common_.engine_type == daqiri::EngineType::GPUNETIO_RDMA,
         "a socket config with engine gpunetio_verbs resolves to the gpunetio_rdma engine");
  expect(daqiri::config_engine_to_string(daqiri::EngineType::GPUNETIO_RDMA) == "gpunetio_verbs",
         "the gpunetio_rdma engine is reported under its user-facing name");
  expect(base.common_.protocol == daqiri::SocketProtocol::ROCE, "base config uses RoCE");
  expect(base.ifs_.size() == 2 && base.ifs_[0].rdma_.mode_ == daqiri::RDMAMode::CLIENT &&
             base.ifs_[1].rdma_.mode_ == daqiri::RDMAMode::SERVER &&
             base.ifs_[1].rdma_.port_ == 4096,
         "client and server interfaces");
  expect(daqiri::validate_network_config(base), "base config is valid");

  for (const auto& unsupported : kUnsupportedCases) {
    std::string yaml = kBaseConfig;
    if (!patch(&yaml, unsupported.from, unsupported.to)) {
      expect(false, std::string(unsupported.name) + ": marker not found in the base config");
      continue;
    }
    daqiri::NetworkConfig config;
    expect(parse(yaml, &config), std::string(unsupported.name) + ": config parses");
    expect(!daqiri::validate_network_config(config),
           std::string(unsupported.name) + ": config is rejected");
  }

  for (const auto& invalid : kInvalidCases) {
    std::string yaml = kBaseConfig;
    if (!patch(&yaml, invalid.from, invalid.to)) {
      expect(false, std::string(invalid.name) + ": marker not found in the base config");
      continue;
    }
    daqiri::NetworkConfig config;
    expect(!parse(yaml, &config), std::string(invalid.name) + ": config is rejected");
  }

  // Buffers are rounded up to the GPU page size, as with the ibverbs RDMA engine
  daqiri::MemoryRegionRequirements requirements;
  expect(daqiri::get_memory_region_requirements(base, requirements) == daqiri::Status::SUCCESS,
         "memory region requirements");
  const auto server = requirements.find("DATA_SERVER");
  expect(server != requirements.end() && server->second.slot_size % 65536 == 0 &&
             server->second.slot_size >= 1048576,
         "memory region slot size");

  return failures == 0 ? 0 : 1;
}
