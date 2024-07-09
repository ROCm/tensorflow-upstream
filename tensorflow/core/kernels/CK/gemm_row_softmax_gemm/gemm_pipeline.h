// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "gemm_row_softmax_gemm_problem.h"
#include "gemm_row_softmax_gemm_pipeline_policy.h"

namespace ck_tile {

template <typename GemmRowSoftmaxGemmProblem_, typename GemmTraits_>
struct BlockGemmPipeline
{
    using Problem = remove_cvref_t<GemmRowSoftmaxGemmProblem_>;
    using Traits  = remove_cvref_t<GemmTraits_>;

    using Policy = GemmRowSoftmaxGemmPolicy;

    using InOutDataType   = remove_cvref_t<typename Problem::InOutDataType>;
    using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;

    using GemmTileShape = remove_cvref_t<typename Problem::GemmTileShape>;

    static constexpr index_t kM    = GemmTileShape::kGemmM;
    static constexpr index_t kN    = GemmTileShape::kGemmN;
    static constexpr index_t kK    = GemmTileShape::kGemmK;
    static constexpr index_t kMaxK = GemmTileShape::kMaxK;

    static constexpr bool kPadSeqLen     = Traits::kPadGemmN;
    static constexpr bool kPadHeadDim    = Traits::kPadGemmM;
    static constexpr bool kPadNewHeadDim = Traits::kPadGemmK;

    static constexpr index_t kAlignmentB1 = 1;
    static constexpr index_t kAlignmentA1 = 1;
    static constexpr index_t kAlignmentD  = 1;

    static constexpr index_t kBlockSize  = Problem::kGemm1BlockSize;
    static constexpr index_t kBlockPerCu = Traits::kGemmBlockPerCu;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::GetGemmKernelSmemSize<Problem>();
    }

    template <typename B1TDramBlockWindowTmp, typename A1DramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto operator()(const B1TDramBlockWindowTmp& b1t_dram_block_window_tmp,
                                        const A1DramBlockWindowTmp& a1_dram_block_window_tmp,
                                        void* smem_ptr) const
    {
        static_assert(std::is_same_v<InOutDataType,
                                     remove_cvref_t<typename B1TDramBlockWindowTmp::DataType>> &&
                          std::is_same_v<InOutDataType,
                                         remove_cvref_t<typename A1DramBlockWindowTmp::DataType>>,
                      "wrong!");

        static_assert(kN == B1TDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kM == A1DramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kMaxK == B1TDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kMaxK == A1DramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // A1 tile in LDS
        InOutDataType* a1_lds_ptr = static_cast<InOutDataType*>(static_cast<void*>(smem_ptr));
        auto a1_lds               = make_tensor_view<address_space_enum::lds>(
            a1_lds_ptr, Policy::template MakeA1LdsBlockDescriptor<Problem>());
        auto a1_lds_window =
            make_tile_window(a1_lds, make_tuple(number<kM>{}, number<kMaxK>{}), {0, 0});

        // Block GEMM
        constexpr auto gemm = Policy::template GetA1B1TBlockGemm<Problem>();

        auto a1_dram_window = make_tile_window(
            a1_dram_block_window_tmp.get_bottom_tensor_view(),
            a1_dram_block_window_tmp.get_window_lengths(),
            a1_dram_block_window_tmp.get_window_origin(),
            Policy::template MakeA1DramTileDistribution<Problem, decltype(gemm)>());

        auto b1t_dram_window = make_tile_window(
            b1t_dram_block_window_tmp.get_bottom_tensor_view(),
            b1t_dram_block_window_tmp.get_window_lengths(),
            b1t_dram_block_window_tmp.get_window_origin(),
            Policy::template MakeB1TDramTileDistribution<Problem, decltype(gemm)>());

        // Load in once a1 from global to lds
        auto a1_tile = load_tile(a1_dram_window);
        store_tile(a1_lds_window, a1_tile);

        using AccBlockTileType = decltype(gemm.MakeCBlockTile());
        auto s_acc             = AccBlockTileType{};

        constexpr auto k_loops = kMaxK / kK;

        auto b1t_tile = load_tile(b1t_dram_window);

        block_sync_lds();

        s_acc = gemm(get_slice_tile(a1_lds_window, sequence<0, 0>{}, sequence<kM, kK>{}),
                     get_slice_tile(b1t_tile, sequence<0, 0>{}, sequence<kN, kK>{}));

        static_for<1, k_loops, 1>{}([&](auto i_k) {
            gemm(s_acc,
                 get_slice_tile(
                     a1_lds_window, sequence<0, i_k * kK>{}, sequence<kM, (i_k + 1) * kK>{}),
                 get_slice_tile(b1t_tile, sequence<0, i_k * kK>{}, sequence<kN, (i_k + 1) * kK>{}));
        });

        tile_elementwise_inout([&](auto& x) { x = (x > 0) ? x : 0; }, s_acc);

        const auto d_tile = cast_tile<InOutDataType>(s_acc);

        return d_tile;
    };
};

} // namespace ck_tile
