// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

struct GemmRowSoftmaxGemmParams {
  const void* b0_ptr;
  const void* a0_ptr;
  const void* mask_ptr;
  void* b1_ptr;
  const void* a1_ptr;
  void* d_ptr;

  ck_tile::index_t num_batch;
  ck_tile::index_t seqlen;
  ck_tile::index_t b0_head_sz;  // head_sz
  ck_tile::index_t b1_head_sz;  // new_head_sz
  ck_tile::index_t b0_batch_stride;
  ck_tile::index_t b0_head_stride;
  ck_tile::index_t
      a0_ld_sz;  // leading dim size or stride for the non-leading dim
  ck_tile::index_t mask_batch_stride;
  ck_tile::index_t b1_batch_stride;
  ck_tile::index_t b1_head_stride;
  ck_tile::index_t
      a1_ld_sz;  // leading dim size or stride for the non-leading dim
  ck_tile::index_t d_batch_stride;
  ck_tile::index_t d_head_stride;
};
