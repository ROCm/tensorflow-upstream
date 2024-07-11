// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/block_gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

struct GemvSoftmaxGemvPolicy {
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetGemvSoftmaxGemvSmemSize() {
    return (MakeB0LdsBlockDescriptor<Problem>().get_element_space_size() +
            MakeB1TLdsBlockDescriptor<Problem>().get_element_space_size()) *
           sizeof(typename Problem::InOutDataType);
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB0() {
    return MakeB0LdsBlockDescriptor<Problem>().get_element_space_size() *
           sizeof(typename Problem::InOutDataType);
  }

  template <typename Problem, typename BlockGemm>
  CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution() {
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    static_assert(
        NWarp == 1,
        "NWarp == 1 is required since dim-N of Gemm0 is dim-K of Gemm1");

    // Gemm0 is restricted to have single warp allocate on dim-N, but Gemm1 is
    // not restricted
    constexpr index_t Gemm1NWarp =
        Problem::GemvSoftmaxGemvTileShape::NumWarpsGemm1 /
        Problem::GemvSoftmaxGemvTileShape::NumWarpsGemm0;

    constexpr index_t kMPerBlock = Problem::GemvSoftmaxGemvTileShape::kM;
    constexpr index_t kKPerBlock = Problem::GemvSoftmaxGemvTileShape::kK0;
    constexpr index_t kMaxK = Problem::GemvSoftmaxGemvTileShape::kMaxK;

    // K2 is equal to Impl::kABKPerLane * kKIterPerWarpGemm
    constexpr index_t K3 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K2 = WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K1 = kKPerBlock / (K2 * K3);
    constexpr index_t K0 = kMaxK / kKPerBlock;
    constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
    constexpr index_t M1 = MWarp;
    constexpr index_t M0 = kMPerBlock / (M2 * M1);
    /*
            constexpr auto a_block_dstr_encoding =
                tile_distribution_encoding<sequence<M0, M1, M2, NWarp>,
                                           tuple<sequence<K0, K1, K2, K3>>,
                                           tuple<sequence<0, 0>, sequence<1,
       0>>, tuple<sequence<1, 3>, sequence<2, 2>>, sequence<1, 1, 1>,
                                           sequence<0, 1, 3>>{};

            constexpr auto a_block_dstr =
       make_static_tile_distribution(a_block_dstr_encoding);

            return a_block_dstr;
    */
    constexpr auto a_block_dstr_encoding = tile_distribution_encoding<
        sequence<Gemm1NWarp>,
        tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2, K3>>,
        tuple<sequence<1, 0>, sequence<2, 1>>,
        tuple<sequence<1, 0>, sequence<2, 2>>, sequence<1, 2, 2, 2>,
        sequence<0, 0, 1, 3>>{};

    constexpr auto a_block_dstr =
        make_static_tile_distribution(a_block_dstr_encoding);

    return a_block_dstr;
  }

  template <typename Problem, typename BlockGemm>
  CK_TILE_HOST_DEVICE static constexpr auto MakeB0DramTileDistribution() {
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    static_assert(
        NWarp == 1,
        "NWarp == 1 is required since dim-N of Gemm0 is dim-K of Gemm1");

    // Gemm0 is restricted to have single warp allocate on dim-N, but Gemm1 is
    // not restricted
    constexpr index_t Gemm1NWarp =
        Problem::GemvSoftmaxGemvTileShape::NumWarpsGemm1 /
        Problem::GemvSoftmaxGemvTileShape::NumWarpsGemm0;

    constexpr index_t kNPerBlock = Problem::GemvSoftmaxGemvTileShape::kN0;
    constexpr index_t kKPerBlock = Problem::GemvSoftmaxGemvTileShape::kK0;

    // K2 is equal to Impl::kABKPerLane * kKIterPerWarpGemm
    constexpr index_t K2 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K0 = kKPerBlock / (K1 * K2);
    constexpr index_t N2 = WG::WarpGemmAttribute::Impl::kBNLane;
    constexpr index_t N1 = NWarp;
    constexpr index_t N0 = kNPerBlock / (N2 * N1);

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<MWarp, Gemm1NWarp>,
            tuple<sequence<N0, N1, N2>, sequence<K0, K1, K2>>,
            tuple<sequence<0, 0, 1>, sequence<2, 1>>,
            tuple<sequence<0, 1, 1>, sequence<1, 2>>, sequence<1, 2, 2>,
            sequence<0, 0, 2>>{});
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeB0LdsBlockDescriptor() {
    using B0DataType = remove_cvref_t<typename Problem::InOutDataType>;

    constexpr index_t kNPerBlock = Problem::GemvSoftmaxGemvTileShape::kN0;
    constexpr index_t kKPerBlock = Problem::GemvSoftmaxGemvTileShape::kK0;
    constexpr index_t kKPack = 8 / sizeof(B0DataType);

    constexpr auto b0_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kKPerBlock / kKPack>{}, number<kNPerBlock>{},
                   number<kKPack>{}),
        make_tuple(number<(kNPerBlock + 1) * kKPack>{}, number<kKPack>{},
                   number<1>{}),
        number<4>{}, number<1>{});

    constexpr auto b0_lds_block_desc = transform_tensor_descriptor(
        b0_lds_block_desc_0,
        make_tuple(make_pass_through_transform(number<kNPerBlock>{}),
                   make_merge_transform(make_tuple(
                       number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return b0_lds_block_desc;
  }

  template <typename Problem, typename BlockGemm>
  CK_TILE_HOST_DEVICE static constexpr auto MakeMaskDramTileDistribution() {
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    static_assert(
        NWarp == 1,
        "NWarp == 1 is required since dim-N of Gemm0 is dim-K of Gemm1");

    // Gemm0 is restricted to have single warp allocate on dim-N, but Gemm1 is
    // not restricted
    constexpr index_t Gemm1NWarp =
        Problem::GemvSoftmaxGemvTileShape::NumWarpsGemm1 /
        Problem::GemvSoftmaxGemvTileShape::NumWarpsGemm0;

    constexpr index_t kMPerBlock = Problem::GemvSoftmaxGemvTileShape::kM;
    constexpr index_t kNPerBlock = Problem::GemvSoftmaxGemvTileShape::kN0;

    constexpr index_t N2 = WG::WarpGemmAttribute::Impl::kCNLane;
    constexpr index_t N1 = NWarp;
    constexpr index_t N0 = kNPerBlock / (N1 * N2);

    constexpr index_t M4 = WG::WarpGemmAttribute::Impl::kCM1PerLane;
    constexpr index_t M3 = WG::WarpGemmAttribute::Impl::kCMLane;
    constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kCM0PerLane;
    constexpr index_t M1 = MWarp;
    constexpr index_t M0 = kMPerBlock / (M1 * M2 * M3 * M4);

    static_assert(N2 == M2 * M3 * M4,
                  "TransposedC WarpGemmAttribute requirements not satisfied!");

    // remember that C warp-tile is transposed by A0B0T BlockGemm
    constexpr auto mask_warp_dstr_encoding =
        tile_distribution_encoding<sequence<N2>, tuple<sequence<M2, M3, M4>>,
                                   tuple<sequence<1, 0>>, tuple<sequence<1, 0>>,
                                   sequence<1, 1>, sequence<0, 2>>{};

    constexpr auto mask_block_outer_dstr_encoding = tile_distribution_encoding<
        sequence<M0, M1, Gemm1NWarp>, tuple<sequence<N0, N1>>,
        tuple<sequence<0, 0, 1>>, tuple<sequence<1, 2, 1>>, sequence<1>,
        sequence<0>>{};

    constexpr auto mask_block_dstr_encoding =
        detail::make_embed_tile_distribution_encoding(
            mask_block_outer_dstr_encoding, mask_warp_dstr_encoding);

    constexpr auto mask_block_dstr =
        make_static_tile_distribution(mask_block_dstr_encoding);

    return mask_block_dstr;
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetAB0BlockGemm() {
    using BlockGemmProblem = BlockGemmPipelineProblem<
        typename Problem::InOutDataType, typename Problem::InOutDataType,
        typename Problem::GemmAccDataType, Problem::kBlockSize,
        TileGemmShape<Problem::GemvSoftmaxGemvTileShape::kM,
                      Problem::GemvSoftmaxGemvTileShape::kN0,
                      Problem::GemvSoftmaxGemvTileShape::kK0>>;

    constexpr auto warp_gemm = []() {
      if constexpr (std::is_same_v<typename Problem::InOutDataType, fp16_t> &&
                    std::is_same_v<typename Problem::GemmAccDataType, float>) {
        return WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
      } else if constexpr (std::is_same_v<typename Problem::InOutDataType,
                                          bf16_t> &&
                           std::is_same_v<typename Problem::GemmAccDataType,
                                          float>) {
        return WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
      }
    }();

    using BlockGemmPolicy = BlockGemmARegBSmemCRegV1CustomPolicy<
        typename Problem::InOutDataType, typename Problem::InOutDataType,
        typename Problem::GemmAccDataType,
        typename Problem::GemvSoftmaxGemvTileShape::Gemm0BlockWarps,
        decltype(warp_gemm)>;

    return BlockGemmARegBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
  }

  template <typename Problem, typename BlockGemm>
  CK_TILE_HOST_DEVICE static constexpr auto MakeB1TDramTileDistribution() {
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t kNPerBlock = Problem::GemvSoftmaxGemvTileShape::kN1;
    constexpr index_t kKPerBlock = Problem::GemvSoftmaxGemvTileShape::kK1;

    // K2 is equal to Impl::kABKPerLane * kKIterPerWarpGemm
    constexpr index_t K2 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K0 = kKPerBlock / (K1 * K2);
    constexpr index_t N2 = WG::WarpGemmAttribute::Impl::kBNLane;
    constexpr index_t N1 = NWarp;
    constexpr index_t N0 = kNPerBlock / (N2 * N1);

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<MWarp>, tuple<sequence<N0, N1, N2>, sequence<K0, K1, K2>>,
            tuple<sequence<0, 1>, sequence<2, 1>>,
            tuple<sequence<0, 1>, sequence<1, 2>>, sequence<1, 2, 2>,
            sequence<0, 0, 2>>{});
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeB1TLdsBlockDescriptor() {
    using B1DataType = remove_cvref_t<typename Problem::InOutDataType>;

    constexpr index_t kNPerBlock = Problem::GemvSoftmaxGemvTileShape::kN1;
    constexpr index_t kKPerBlock = Problem::GemvSoftmaxGemvTileShape::kK1;
    constexpr index_t kKPack = 8 / sizeof(B1DataType);

    constexpr auto b1t_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kKPerBlock / kKPack>{}, number<kNPerBlock>{},
                   number<kKPack>{}),
        make_tuple(number<(kNPerBlock + 1) * kKPack>{}, number<kKPack>{},
                   number<1>{}),
        number<4>{}, number<1>{});

    constexpr auto b1t_lds_block_desc = transform_tensor_descriptor(
        b1t_lds_block_desc_0,
        make_tuple(make_pass_through_transform(number<kNPerBlock>{}),
                   make_merge_transform(make_tuple(
                       number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return b1t_lds_block_desc;
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetPB1TBlockGemm() {
    using BlockGemmProblem = BlockGemmPipelineProblem<
        typename Problem::InOutDataType, typename Problem::InOutDataType,
        typename Problem::GemmAccDataType, Problem::kBlockSize,
        TileGemmShape<Problem::GemvSoftmaxGemvTileShape::kM,
                      Problem::GemvSoftmaxGemvTileShape::kN1,
                      Problem::GemvSoftmaxGemvTileShape::kK1>>;

    constexpr auto warp_gemm = []() {
      if constexpr (std::is_same_v<typename Problem::InOutDataType, fp16_t> &&
                    std::is_same_v<typename Problem::GemmAccDataType, float>) {
        return WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
      } else if constexpr (std::is_same_v<typename Problem::InOutDataType,
                                          bf16_t> &&
                           std::is_same_v<typename Problem::GemmAccDataType,
                                          float>) {
        return WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
      }
    }();

    using BlockGemmPolicy = BlockGemmARegBSmemCRegV1CustomPolicy<
        typename Problem::InOutDataType, typename Problem::InOutDataType,
        typename Problem::GemmAccDataType,
        typename Problem::GemvSoftmaxGemvTileShape::Gemm1BlockWarps,
        decltype(warp_gemm)>;

    return BlockGemmARegBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
  }
};

}  // namespace ck_tile
