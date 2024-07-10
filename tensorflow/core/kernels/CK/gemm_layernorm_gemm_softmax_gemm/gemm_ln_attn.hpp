// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "gemm_ln_attn_kernel.hpp"

namespace ck_tile {
// This class is used for codegen pattern matching
enum class BlockGemmLnAttnPipelineEnum
{
    QWKV_LDS = 0,
};

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
auto gemm_ln_attn_create_kargs_and_grids(gemm_ln_attn_args args)
{
    auto kargs = GemmLnAttnKernel::MakeKargs(args.q_ptr,
                                             args.w_ptr,
                                             args.bias_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             args.gamma_ptr,
                                             args.beta_ptr,
                                             args.keymask_ptr,
                                             args.o_ptr,
                                             args.K,
                                             args.M,
                                             args.N0,
                                             args.N1,
                                             args.N2,
                                             args.batch_kv,
                                             args.nhead,
                                             args.lrelu_alpha,
                                             args.do_leaky_relu);
    // ignore do_layer_norm, do_query_mask;

    dim3 grids = GemmLnAttnKernel::GridSize(args.batch_kv, args.nhead, args.M);

    return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to
// instantiate kernel.
template <bool kPadBatchQ_ /* padding for seqlen_q */,
          bool kPadHeadDimQ_ /* paddding for hdim_q */,
          bool kPadQWGemmK_ /* padding for qwgemm_k */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimV_ /* paddding for hdim_v */,
          index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct TileGemmLnAttnTraits
{
    static constexpr bool kPadBatchQ     = kPadBatchQ_;
    static constexpr bool kPadHeadDimQ   = kPadHeadDimQ_;
    static constexpr bool kPadQWGemmK    = kPadQWGemmK_;
    static constexpr bool kPadSeqLenK    = kPadSeqLenK_;
    static constexpr bool kPadHeadDimV   = kPadHeadDimV_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

template <typename BlockTile_, // sequence<...
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_>
struct TileGemmLnGemmShape
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;

    static constexpr index_t NumWarps =
        reduce_on_sequence(Gemm0BlockWarps{}, multiplies{}, number<1>{});

    static_assert(NumWarps == reduce_on_sequence(Gemm1BlockWarps{}, multiplies{}, number<1>{}));

    static constexpr index_t kM0 = BlockTile::at(number<0>{}); // tile size along q batch
    static constexpr index_t kN0 = BlockTile::at(number<1>{}); // tile size along qk head_dim
    static constexpr index_t kK0 = BlockTile::at(number<2>{}); // tile size along qw gemm unroll
    static constexpr index_t kN1 = BlockTile::at(number<3>{}); // tile size along k seqlen
    static constexpr index_t kK1 = BlockTile::at(number<4>{}); // tile size along rk gemm unroll
};

// This is the public API, will be generated by script.
struct gemm_ln_attn_traits {
  int hdim_q;
  int hdim_v;
  std::string data_type;
  // TODO: padding check is inside this api
};

template <typename BlockTile_, // sequence<...
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_,
          typename Gemm2BlockWarps_,
          typename Gemm2WarpTile_>
struct TileGemmLnAttnShape
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;
    using Gemm2BlockWarps = remove_cvref_t<Gemm2BlockWarps_>;
    using Gemm2WarpTile   = remove_cvref_t<Gemm2WarpTile_>;

    static constexpr index_t NumWarps =
        reduce_on_sequence(Gemm0BlockWarps{}, multiplies{}, number<1>{});

    static_assert(NumWarps == reduce_on_sequence(Gemm1BlockWarps{}, multiplies{}, number<1>{}));

    static constexpr index_t kM0 = BlockTile::at(number<0>{}); // tile size along q batch
    static constexpr index_t kN0 = BlockTile::at(number<1>{}); // tile size along qk head_dim
    static constexpr index_t kK0 = BlockTile::at(number<2>{}); // tile size along qw gemm unroll
    static constexpr index_t kN1 = BlockTile::at(number<3>{}); // tile size along k seqlen
    static constexpr index_t kK1 = BlockTile::at(number<4>{}); // tile size along rk gemm unroll
    static constexpr index_t kN2 = BlockTile::at(number<5>{}); // tile size along v head_dim
    static constexpr index_t kK2 = BlockTile::at(number<6>{}); // tile size along pv gemm unroll
};

template <typename QDataType_,
          typename WDataType_,
          typename S0accDataType_,
          typename BiasDataType_,
          typename GammaDataType_,
          typename BetaDataType_,
          typename RDataType_,
          typename KDataType_,
          typename S1accDataType_,
          typename KeyMaskDataType_,
          typename SMPLComputeDataType_,
          typename PDataType_,
          typename VDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockGemmLnAttnShape_,
          typename Traits_>
struct BlockGemmLnAttnPipelineProblem
{
    using QDataType            = remove_cvref_t<QDataType_>;
    using WDataType            = remove_cvref_t<WDataType_>;
    using S0accDataType        = remove_cvref_t<S0accDataType_>;
    using BiasDataType         = remove_cvref_t<BiasDataType_>;
    using GammaDataType        = remove_cvref_t<GammaDataType_>;
    using BetaDataType         = remove_cvref_t<BetaDataType_>;
    using RDataType            = remove_cvref_t<RDataType_>;
    using KDataType            = remove_cvref_t<KDataType_>;
    using S1accDataType        = remove_cvref_t<S1accDataType_>;
    using KeyMaskDataType      = remove_cvref_t<KeyMaskDataType_>;
    using SMPLComputeDataType  = remove_cvref_t<SMPLComputeDataType_>;
    using PDataType            = remove_cvref_t<PDataType_>;
    using VDataType            = remove_cvref_t<VDataType_>;
    using OaccDataType         = remove_cvref_t<OaccDataType_>;
    using ODataType            = remove_cvref_t<ODataType_>;
    using BlockGemmLnAttnShape = remove_cvref_t<BlockGemmLnAttnShape_>;
    using Traits               = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize = BlockGemmLnAttnShape::NumWarps * get_warp_size();

    // attributes from traits
    static constexpr bool kPadBatchQ     = Traits::kPadBatchQ;
    static constexpr bool kPadHeadDimQ   = Traits::kPadHeadDimQ;
    static constexpr bool kPadQWGemmK    = Traits::kPadQWGemmK;
    static constexpr bool kPadSeqLenK    = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimV   = Traits::kPadHeadDimV;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
};

template <typename BlockGemmLnAttnShape_>
struct GemmLnAttnTilePartitioner
{
    using BlockGemmLnAttnShape = ck_tile::remove_cvref_t<BlockGemmLnAttnShape_>;

    static constexpr ck_tile::index_t kM0 = BlockGemmLnAttnShape::kM0;

    static constexpr const char* name = "hbs";

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_q_)
    {
        // TODO: this may need tuning
        return dim3(nhead_, batch_size_, ck_tile::integer_divide_ceil(seqlen_q_, kM0));
    }

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*seqlen_q*/)
    {
        const index_t i_block = blockIdx.z;
        const index_t i_nhead = blockIdx.x;
        const index_t i_batch = blockIdx.y;

        return ck_tile::make_tuple(i_block, i_nhead, i_batch);
    }
};

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDimQ_,
          ck_tile::index_t HDimV_,
          typename DataType_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kN2_,
          ck_tile::index_t kK2_,
          ck_tile::BlockGemmLnAttnPipelineEnum GemmLnAttnPipelineEnum_,
          bool kPadBatchQ_,
          bool kPadQWK_,
          bool kPadD_,
          bool kPadSK_,
          bool kPadDv_>
struct gemm_ln_attn_traits_
{
    static constexpr ck_tile::index_t HDimQ = HDimQ_;
    static constexpr ck_tile::index_t HDimV = HDimV_;
    using DataType                          = ck_tile::remove_cvref_t<DataType_>;
    static constexpr ck_tile::index_t kM0   = kM0_;
    static constexpr ck_tile::index_t kN0   = kN0_;
    static constexpr ck_tile::index_t kK0   = kK0_;
    static constexpr ck_tile::index_t kN1   = kN1_;
    static constexpr ck_tile::index_t kK1   = kK1_;
    static constexpr ck_tile::index_t kN2   = kN2_;
    static constexpr ck_tile::index_t kK2   = kK2_;
    static constexpr bool kPadBatchQ        = kPadBatchQ_;
    static constexpr bool kPadD             = kPadD_;
    static constexpr bool kPadQWK           = kPadQWK_;
    static constexpr bool kPadSK            = kPadSK_;
    static constexpr bool kPadDv            = kPadDv_;
};

template <typename Traits_>
float gemm_ln_attn_(const ck_tile::stream_config&, gemm_ln_attn_args);

float gemm_ln_attn(gemm_ln_attn_traits, gemm_ln_attn_args, const ck_tile::stream_config&);


} // namespace ck_tile
