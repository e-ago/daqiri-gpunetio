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

// Checks, without a GPU or NIC, how the configuration parser and the shared validation handle the
// gpunetio_verbs engine. Nothing here creates the engine or opens a device.

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

#include <daqiri/daqiri.h>

#include "src/engine.h"

namespace {

const char* const kBaseConfig = R"yaml(%YAML 1.2
---
daqiri:
  cfg:
    version: 1
    stream_type: "raw"
    engine: "gpunetio_verbs"
    master_core: 0
    log_level: "info"
    memory_regions:
    - name: "DATA_TX"
      kind: "device"
      affinity: 0
      num_bufs: 1024
      buf_size: 2048
    - name: "DATA_RX"
      kind: "device"
      affinity: 0
      num_bufs: 1024
      buf_size: 2000
    interfaces:
    - name: "port0"
      address: "0000:00:00.0"
      tx:
        queues:
        - name: "tx_q_0"
          id: 0
          cpu_core: 1
          batch_size: 256
          memory_regions:
          - "DATA_TX"
      rx:
        queues:
        - name: "rx_q_0"
          id: 0
          cpu_core: 2
          batch_size: 256
          memory_regions:
          - "DATA_RX"
        flows:
        - name: "flow_0"
          id: 1
          action:
            type: queue
            id: 0
          match:
            udp_src: 4096
            udp_dst: 4096
)yaml";

struct Case {
  const char* name;
  const char* from;
  const char* to;
};

const char* const kFlowMatch = "            udp_src: 4096\n            udp_dst: 4096\n";

// A second RX queue on a region of its own
const char* const kSecondRxQueue =
    "          - \"DATA_RX\"\n        - name: \"rx_q_1\"\n          id: 1\n          cpu_core: 3\n"
    "          batch_size: 256\n          memory_regions:\n          - \"DATA_RX_1\"\n";
const char* const kSecondRxRegion = "    interfaces:";
const char* const kSecondRxRegionTo =
    "    - name: \"DATA_RX_1\"\n      kind: \"device\"\n      affinity: 0\n      num_bufs: 1024\n"
    "      buf_size: 2048\n    interfaces:";

// Configurations the parser accepts and the shared validation rejects
const std::vector<Case> kUnsupportedCases = {
    {"header-data split", "          - \"DATA_RX\"\n",
     "          - \"DATA_RX\"\n          - \"DATA_TX\"\n"},
    {"host memory", "    - name: \"DATA_TX\"\n      kind: \"device\"\n",
     "    - name: \"DATA_TX\"\n      kind: \"host\"\n"},
    {"software loopback", "    log_level: \"info\"\n",
     "    log_level: \"info\"\n    loopback: \"sw\"\n"},
    {"TX offload", "          - \"DATA_TX\"\n",
     "          - \"DATA_TX\"\n          offloads:\n          - \"tx_eth_src\"\n"},
    {"packet pacing", "          - \"DATA_TX\"\n",
     "          - \"DATA_TX\"\n          pacing_mbps: 1000\n"},
    {"accurate send", "      tx:\n", "      tx:\n        accurate_send: true\n"},
    {"hardware timestamps", "      rx:\n", "      rx:\n        hardware_timestamps: true\n"},
    {"dynamic RX flows", "      rx:\n", "      rx:\n        dynamic_flow_capacity: 16\n"},
    {"eCPRI flow", kFlowMatch, "            ecpri: {}\n"},
    // IPv4 total length needs a flex parser
    {"IPv4 length match", kFlowMatch, "            ipv4_len: 1024\n"},
    // The NIC tags each packet with its flow ID in 24 bits
    {"flow ID over 24 bits", "          id: 1\n", "          id: 16777216\n"},
    // The region of an RX queue is its receive ring
    {"shared RX region", "          - \"DATA_RX\"\n", "          - \"DATA_TX\"\n"},
    // A TX queue hands out the packet slots of its whole region
    {"shared TX region", "          - \"DATA_TX\"\n      rx:\n",
     "          - \"DATA_TX\"\n        - name: \"tx_q_1\"\n          id: 1\n          cpu_core: 3\n"
     "          batch_size: 256\n          memory_regions:\n          - \"DATA_TX\"\n      rx:\n"},
    {"ring smaller than a batch", "      num_bufs: 1024\n      buf_size: 2000\n",
     "      num_bufs: 128\n      buf_size: 2000\n"},
    {"TX batch bigger than its region", "      num_bufs: 1024\n      buf_size: 2048\n",
     "      num_bufs: 128\n      buf_size: 2048\n"},
    // The kernels and the queue rings run on one GPU
    {"queues on two GPUs", "    - name: \"DATA_RX\"\n      kind: \"device\"\n      affinity: 0",
     "    - name: \"DATA_RX\"\n      kind: \"device\"\n      affinity: 1"},
};

// Configurations the parser rejects
const std::vector<Case> kInvalidCases = {
    // As with "rdma" for "ibverbs", the internal name of the RoCE engine is not valid in a config
    {"internal RoCE engine name", "    engine: \"gpunetio_verbs\"",
     "    engine: \"gpunetio_rdma\""},
    {"hardware loopback", "    log_level: \"info\"\n",
     "    log_level: \"info\"\n    loopback: \"hw\"\n"},
    {"direct RX polling", "          cpu_core: 2\n          batch_size: 256\n",
     "          poll_mode: \"direct\"\n"},
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

// The base config with count flows, flow IDs 1 to count, all to RX queue 0
std::string with_flows(int count, bool isolation) {
  std::string yaml = kBaseConfig;
  const std::string flows = "        flows:\n";
  const size_t pos = yaml.find(flows);
  std::string list = flows;
  for (int i = 1; i <= count; i++) {
    list += "        - name: \"flow_" + std::to_string(i) +
            "\"\n          id: " + std::to_string(i) +
            "\n          action:\n            type: queue\n            id: 0\n          match:\n"
            "            udp_dst: " +
            std::to_string(4000 + i) + "\n";
  }
  yaml.replace(pos, std::string::npos, list);
  if (isolation) {
    yaml.replace(yaml.find("      rx:\n"), 10, "      rx:\n        flow_isolation: true\n");
  }
  return yaml;
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
  expect(base.common_.engine_type == daqiri::EngineType::GPUNETIO_VERBS,
         "base config resolves to the gpunetio_verbs engine");
  expect(daqiri::validate_network_config(base), "base config is valid");

  // Supported variants
  {
    std::string yaml = kBaseConfig;
    expect(patch(&yaml, "          - \"DATA_RX\"\n", kSecondRxQueue) &&
               patch(&yaml, kSecondRxRegion, kSecondRxRegionTo) &&
               patch(&yaml, "            type: queue\n            id: 0\n",
                     "            type: queue\n            ids: [0, 1]\n"),
           "RSS markers");
    daqiri::NetworkConfig config;
    expect(parse(yaml, &config) && daqiri::validate_network_config(config),
           "RSS over two RX queues is valid");
  }
  {
    std::string yaml = kBaseConfig;
    expect(
        patch(&yaml, kFlowMatch, "            ethernet:\n              dst: 02:00:00:00:00:01\n"),
        "Ethernet match marker");
    daqiri::NetworkConfig config;
    expect(parse(yaml, &config) && daqiri::validate_network_config(config),
           "an Ethernet match is valid");
  }
  {
    std::string yaml = kBaseConfig;
    expect(patch(&yaml, "      kind: \"device\"", "      kind: \"host_pinned\"") &&
               patch(&yaml, "      kind: \"device\"", "      kind: \"host_pinned\""),
           "host_pinned markers");
    daqiri::NetworkConfig config;
    expect(parse(yaml, &config) && daqiri::validate_network_config(config),
           "host_pinned memory regions are valid");
  }

  // The NIC root table has 16 priorities: the RX flows, plus the rule for unmatched packets when
  // flow_isolation is false
  for (const auto& [flows, isolation, valid] : std::vector<std::tuple<int, bool, bool>>{
           {15, false, true}, {16, false, false}, {16, true, true}, {17, true, false}}) {
    daqiri::NetworkConfig config;
    const std::string what = std::to_string(flows) + " flows" + (isolation ? " isolated" : "");
    expect(parse(with_flows(flows, isolation), &config), what + ": config parses");
    expect(daqiri::validate_network_config(config) == valid,
           what + (valid ? ": config is valid" : ": config is rejected"));
  }

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

  // Slots are rounded up to a multiple of 64 bytes
  daqiri::MemoryRegionRequirements requirements;
  expect(daqiri::get_memory_region_requirements(base, requirements) == daqiri::Status::SUCCESS,
         "memory region requirements");
  const auto rx = requirements.find("DATA_RX");
  expect(rx != requirements.end() && rx->second.slot_size == 2048 && rx->second.num_bufs == 1024,
         "memory region slot size");

  return failures == 0 ? 0 : 1;
}
