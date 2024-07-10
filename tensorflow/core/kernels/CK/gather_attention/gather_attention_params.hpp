// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

struct GatherAttentionParams {
  const void* a_ptr;
  const void* b0_ptr;
  const void* mask_ptr;
  void* indices_ptr;
  const void* b1_ptr;
  void* d_ptr;

  ck_tile::index_t num_batch;
  ck_tile::index_t num_index;
  ck_tile::index_t seqlen;
  ck_tile::index_t num_head;
  ck_tile::index_t head_sz;
  ck_tile::index_t a_batch_stride;
  ck_tile::index_t a_nhead_stride;
  ck_tile::index_t b0_batch_stride;
  ck_tile::index_t b0_seq_stride;
  ck_tile::index_t b0_nhead_stride;
  ck_tile::index_t mask_batch_stride;
  ck_tile::index_t b1_batch_stride;
  ck_tile::index_t b1_seq_stride;
  ck_tile::index_t b1_nhead_stride;
  ck_tile::index_t d_batch_stride;
  ck_tile::index_t d_nhead_stride;
};
