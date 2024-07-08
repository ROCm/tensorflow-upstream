// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm_ln_attn.hpp"

template <typename DataType>
struct GemmLnAttnTypeConfig;

template <>
struct GemmLnAttnTypeConfig<ck_tile::half_t> {
  using QDataType = ck_tile::half_t;
  using WDataType = ck_tile::half_t;
  using S0accDataType = float;  // data type for first gemm accumulation
  using BiasDataType = ck_tile::half_t;
  using GammaDataType = ck_tile::half_t;
  using BetaDataType = ck_tile::half_t;

  using RDataType = ck_tile::half_t;  // data type for A matrix of second gemm
  using KDataType = ck_tile::half_t;
  using S1accDataType = float;  // data type for second gemm accumulation
  using KeyMaskDataType = ck_tile::half_t;
  using SMPLComputeDataType = float;  // data type for reduction, softmax

  using PDataType = ck_tile::half_t;  // data type for A matrix of third gemm
  using VDataType = ck_tile::half_t;
  using OaccDataType = float;  // data type for third gemm accumulation
  using ODataType = ck_tile::half_t;
};

// Host API
struct gemm_ln_attn_args {
  const void* q_ptr;
  const void* w_ptr;
  const void* bias_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* gamma_ptr;
  const void* beta_ptr;
  const void* keymask_ptr;
  void* o_ptr;
  ck_tile::index_t K;
  ck_tile::index_t M;
  ck_tile::index_t N0;
  ck_tile::index_t N1;
  ck_tile::index_t N2;
  ck_tile::index_t batch_kv;
  ck_tile::index_t nhead;
  float lrelu_alpha;
  bool do_layer_norm;
  bool do_leaky_relu;
  bool do_query_mask;
};

template <typename GemmLnAttnKernel>
auto gemm_ln_attn_create_kargs_and_grids(gemm_ln_attn_args args) {
  auto kargs = GemmLnAttnKernel::MakeKargs(
      args.q_ptr, args.w_ptr, args.bias_ptr, args.k_ptr, args.v_ptr,
      args.gamma_ptr, args.beta_ptr, args.keymask_ptr, args.o_ptr, args.K,
      args.M, args.N0, args.N1, args.N2, args.batch_kv, args.nhead,
      args.lrelu_alpha, args.do_leaky_relu);
  // ignore do_layer_norm, do_query_mask;

  dim3 grids = GemmLnAttnKernel::GridSize(args.batch_kv, args.nhead, args.M);

  return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to
// instantiate kernel
template <ck_tile::index_t HDimQ_, ck_tile::index_t HDimV_, typename DataType_,
          ck_tile::index_t kM0_, ck_tile::index_t kN0_, ck_tile::index_t kK0_,
          ck_tile::index_t kN1_, ck_tile::index_t kK1_, ck_tile::index_t kN2_,
          ck_tile::index_t kK2_,
          ck_tile::BlockGemmLnAttnPipelineEnum GemmLnAttnPipelineEnum_,
          bool kPadBatchQ_, bool kPadQWK_, bool kPadD_, bool kPadSK_,
          bool kPadDv_>
struct gemm_ln_attn_traits_ {
  static constexpr ck_tile::index_t HDimQ = HDimQ_;
  static constexpr ck_tile::index_t HDimV = HDimV_;
  using DataType = ck_tile::remove_cvref_t<DataType_>;
  static constexpr ck_tile::index_t kM0 = kM0_;
  static constexpr ck_tile::index_t kN0 = kN0_;
  static constexpr ck_tile::index_t kK0 = kK0_;
  static constexpr ck_tile::index_t kN1 = kN1_;
  static constexpr ck_tile::index_t kK1 = kK1_;
  static constexpr ck_tile::index_t kN2 = kN2_;
  static constexpr ck_tile::index_t kK2 = kK2_;
  static constexpr bool kPadBatchQ = kPadBatchQ_;
  static constexpr bool kPadD = kPadD_;
  static constexpr bool kPadQWK = kPadQWK_;
  static constexpr bool kPadSK = kPadSK_;
  static constexpr bool kPadDv = kPadDv_;
};