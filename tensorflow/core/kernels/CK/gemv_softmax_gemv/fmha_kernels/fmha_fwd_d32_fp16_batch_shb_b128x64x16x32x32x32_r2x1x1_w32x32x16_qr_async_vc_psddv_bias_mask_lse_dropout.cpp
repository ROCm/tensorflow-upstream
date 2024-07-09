// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

// auto generated by generate.py
#include "tensorflow/core/kernels/CK/gemv_softmax_gemv/fmha_fwd.h"

using fmha_dtype_0 = ck_tile::fp16_t;

using fmha_block_tile_0 = ck_tile::sequence<128, 64, 16, 32, 32, 32>;
using fmha_block_warps_0 = ck_tile::sequence<2, 1, 1>;
using fmha_warp_tile_0 = ck_tile::sequence<32, 32, 16>;

using fmha_shape_0 = ck_tile::TileFmhaShape<fmha_block_tile_0,
                                      fmha_block_warps_0,
                                      fmha_warp_tile_0,
                                      fmha_block_warps_0,
                                      fmha_warp_tile_0,
                                      false>;

using fmha_trait_0 = ck_tile::TileFmhaTraits<true,
                                                    false,
                                                    true,
                                                    true,
                                                    ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                                                    false,
                                                    true,
                                                    true,
                                                    false,
                                                    -1>;
using fmha_mask_0 = ck_tile::SimplifiedGenericAttentionMask<true>;

using fmha_pipeline_problem_0 = ck_tile::BlockFmhaPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype_0>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::RandValOutputDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_0>::ODataType,
    fmha_shape_0,
    false,
    fmha_mask_0,
    fmha_trait_0>;

using fmha_pipeline_0 = ck_tile::BlockFmhaPipelineQRKSVSAsync<
    fmha_pipeline_problem_0>;

using fmha_epilogue_0 =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<ck_tile::fp16_t>::OaccDataType,
                                           typename FmhaFwdTypeConfig<ck_tile::fp16_t>::ODataType,
                                           true, true>>;

using fmha_kernel_0 =
    ck_tile::FmhaFwdKernel<ck_tile::FmhaFwdTilePartitioner_SHB<fmha_shape_0>,
                  fmha_pipeline_0,
                  fmha_epilogue_0>;

using trait_0 = fmha_fwd_traits_<32, ck_tile::fp16_t, false,128, 64, 16, 32, 32, 32, false,
                        ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC, fmha_mask_0, ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS, true, true, false, true, false, true, true>;

#include <iostream>

template<>
float fmha_fwd_<trait_0>(const ck_tile::stream_config& s, fmha_fwd_args a)
{
    using k_ = fmha_kernel_0;
    if(s.log_level_ > 0)
        std::cout << ", " << k_::GetName() << std::flush;
    auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks             = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{}, grids, blocks, 0, kargs));
}
