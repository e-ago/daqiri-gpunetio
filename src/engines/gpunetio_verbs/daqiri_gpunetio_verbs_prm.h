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

// mlx5 PRM command layouts the gpunetio_verbs engine needs to steer packets to its DOCA Verbs
// receive queues with DevX: a transport domain, TIRs, RQTs and the outer-header match set of the
// mlx5dv_dr rules. The layouts are the ones of the ibverbs engine (src/engines/ibverbs/
// mlx5_prm_min.h), copied from DPDK's drivers/common/mlx5/mlx5_prm.h. Each "1 byte per bit"
// struct must keep the global tag mlx5_ifc_<typ>_bits that the DEVX_SET/DEVX_GET macros of
// <infiniband/mlx5dv.h> expand to.

#pragma once

#include <cstdint>

enum {
  MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN = 0x816,
  MLX5_CMD_OP_CREATE_TIR = 0x900,
  MLX5_CMD_OP_CREATE_RQT = 0x916,
};

enum {
  MLX5_TIRC_DISP_TYPE_DIRECT = 0x0,
  MLX5_TIRC_DISP_TYPE_INDIRECT = 0x1,
};
enum {
  MLX5_RX_HASH_FN_NONE = 0x0,
  MLX5_RX_HASH_FN_TOEPLITZ = 0x2,
};
enum {
  MLX5_L3_PROT_TYPE_IPV4 = 0x0,
  MLX5_L4_PROT_TYPE_UDP = 0x1,
  MLX5_HASH_FIELD_SEL_SRC_IP = 1 << 0,
  MLX5_HASH_FIELD_SEL_DST_IP = 1 << 1,
  MLX5_HASH_FIELD_SEL_L4_SPORT = 1 << 2,
  MLX5_HASH_FIELD_SEL_L4_DPORT = 1 << 3,
};

struct mlx5_ifc_alloc_transport_domain_in_bits {
  uint8_t opcode[0x10];
  uint8_t reserved_at_10[0x10];
  uint8_t reserved_at_20[0x10];
  uint8_t op_mod[0x10];
  uint8_t reserved_at_40[0x40];
};

struct mlx5_ifc_alloc_transport_domain_out_bits {
  uint8_t status[0x8];
  uint8_t reserved_at_8[0x18];
  uint8_t syndrome[0x20];
  uint8_t reserved_at_40[0x8];
  uint8_t transport_domain[0x18];
  uint8_t reserved_at_60[0x20];
};

struct mlx5_ifc_rqtc_bits {
  uint8_t reserved_at_0[0xa0];
  uint8_t reserved_at_a0[0x5];
  uint8_t list_q_type[0x3];
  uint8_t reserved_at_a8[0x8];
  uint8_t rqt_max_size[0x10];
  uint8_t rq_vhca_id_format[0x1];
  uint8_t reserved_at_c1[0xf];
  uint8_t rqt_actual_size[0x10];
  uint8_t reserved_at_e0[0x6a0];
  // Flexible array of 32-bit entries (8 reserved bits + 24-bit RQN). Command buffers allocate
  // additional entries after this sentinel.
  uint8_t rq_num[1][0x20];
};

struct mlx5_ifc_create_rqt_in_bits {
  uint8_t opcode[0x10];
  uint8_t uid[0x10];
  uint8_t reserved_at_20[0x10];
  uint8_t op_mod[0x10];
  uint8_t reserved_at_40[0xc0];
  struct mlx5_ifc_rqtc_bits rqt_context;
};

struct mlx5_ifc_create_rqt_out_bits {
  uint8_t status[0x8];
  uint8_t reserved_at_8[0x18];
  uint8_t syndrome[0x20];
  uint8_t reserved_at_40[0x8];
  uint8_t rqtn[0x18];
  uint8_t reserved_at_60[0x20];
};

struct mlx5_ifc_rx_hash_field_select_bits {
  uint8_t l3_prot_type[0x1];
  uint8_t l4_prot_type[0x1];
  uint8_t selected_fields[0x1e];
};

struct mlx5_ifc_tirc_bits {
  uint8_t reserved_at_0[0x20];
  uint8_t disp_type[0x4];
  uint8_t reserved_at_24[0x1c];
  uint8_t reserved_at_40[0x40];
  uint8_t reserved_at_80[0x4];
  uint8_t lro_timeout_period_usecs[0x10];
  uint8_t lro_enable_mask[0x4];
  uint8_t lro_max_msg_sz[0x8];
  uint8_t reserved_at_a0[0x40];
  uint8_t reserved_at_e0[0x8];
  uint8_t inline_rqn[0x18];
  uint8_t rx_hash_symmetric[0x1];
  uint8_t reserved_at_101[0x1];
  uint8_t tunneled_offload_en[0x1];
  uint8_t reserved_at_103[0x5];
  uint8_t indirect_table[0x18];
  uint8_t rx_hash_fn[0x4];
  uint8_t reserved_at_124[0x2];
  uint8_t self_lb_block[0x2];
  uint8_t transport_domain[0x18];
  uint8_t rx_hash_toeplitz_key[10][0x20];
  struct mlx5_ifc_rx_hash_field_select_bits rx_hash_field_selector_outer;
  struct mlx5_ifc_rx_hash_field_select_bits rx_hash_field_selector_inner;
  uint8_t reserved_at_2c0[0x4c0];
};

struct mlx5_ifc_create_tir_in_bits {
  uint8_t opcode[0x10];
  uint8_t uid[0x10];
  uint8_t reserved_at_20[0x10];
  uint8_t op_mod[0x10];
  uint8_t reserved_at_40[0xc0];
  struct mlx5_ifc_tirc_bits ctx;
};

struct mlx5_ifc_create_tir_out_bits {
  uint8_t status[0x8];
  uint8_t reserved_at_8[0x18];
  uint8_t syndrome[0x20];
  uint8_t reserved_at_40[0x8];
  uint8_t tirn[0x18];
  uint8_t reserved_at_60[0x20];
};

// Layer 2-4 outer-header match set. mlx5dv_dr reads the 64-byte match buffer as this struct when
// match_criteria_enable selects the outer headers.
struct mlx5_ifc_ipv4_layout_bits {
  uint8_t reserved_at_0[0x60];
  uint8_t ipv4[0x20];
};

struct mlx5_ifc_ipv6_layout_bits {
  uint8_t ipv6[16 * 0x8];
};

union mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits {
  struct mlx5_ifc_ipv6_layout_bits ipv6_layout;
  struct mlx5_ifc_ipv4_layout_bits ipv4_layout;
  uint8_t reserved_at_0[0x80];
};

struct mlx5_ifc_fte_match_set_lyr_2_4_bits {
  uint8_t smac_47_16[0x20];
  uint8_t smac_15_0[0x10];
  uint8_t ethertype[0x10];
  uint8_t dmac_47_16[0x20];
  uint8_t dmac_15_0[0x10];
  uint8_t first_prio[0x3];
  uint8_t first_cfi[0x1];
  uint8_t first_vid[0xc];
  uint8_t ip_protocol[0x8];
  uint8_t ip_dscp[0x6];
  uint8_t ip_ecn[0x2];
  uint8_t cvlan_tag[0x1];
  uint8_t svlan_tag[0x1];
  uint8_t frag[0x1];
  uint8_t ip_version[0x4];
  uint8_t tcp_flags[0x9];
  uint8_t tcp_sport[0x10];
  uint8_t tcp_dport[0x10];
  uint8_t reserved_at_c0[0x10];
  uint8_t ipv4_ihl[0x4];
  uint8_t l3_ok[0x1];
  uint8_t l4_ok[0x1];
  uint8_t ipv4_checksum_ok[0x1];
  uint8_t l4_checksum_ok[0x1];
  uint8_t ip_ttl_hoplimit[0x8];
  uint8_t udp_sport[0x10];
  uint8_t udp_dport[0x10];
  union mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits src_ipv4_src_ipv6;
  union mlx5_ifc_ipv6_layout_ipv4_layout_auto_bits dst_ipv4_dst_ipv6;
};

enum {
  MLX5_DR_MATCH_CRITERIA_OUTER = 1 << 0,  // match_criteria_enable: outer_headers
  MLX5_ETHERTYPE_IPV4 = 0x0800,
  MLX5_IP_PROTOCOL_UDP = 0x11,
};
