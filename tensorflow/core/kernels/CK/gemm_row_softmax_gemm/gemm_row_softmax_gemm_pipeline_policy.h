// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/block_gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

struct GemmRowSoftmaxGemmPolicy {
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t
  GetGemmRowSoftmaxKernelSmemSize() {
    return MakeA0LdsBlockDescriptor<Problem>().get_element_space_size() *
           sizeof(typename Problem::InOutDataType);
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetGemmKernelSmemSize() {
    return MakeA1LdsBlockDescriptor<Problem>().get_element_space_size() *
           sizeof(typename Problem::InOutDataType);
  }

  template <typename Problem, typename BlockGemm>
  CK_TILE_HOST_DEVICE static constexpr auto MakeB0TDramTileDistribution() {
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t kNPerBlock = Problem::GemmRowSoftmaxTileShape::kGemmN;
    constexpr index_t kMaxK = Problem::GemmRowSoftmaxTileShape::kMaxK;

    // K2 is equal to Impl::kABKPerLane * kKIter
    constexpr index_t K2 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K0 = kMaxK / (K1 * K2);
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

  template <typename Problem, typename BlockGemm>
  CK_TILE_HOST_DEVICE static constexpr auto MakeA0DramTileDistribution() {
    using A0DataType = remove_cvref_t<typename Problem::InOutDataType>;

    constexpr index_t kBlockSize = Problem::kGemm0BlockSize;
    constexpr index_t kMPerBlock = Problem::GemmRowSoftmaxTileShape::kGemmM;
    constexpr index_t kKPerBlock = Problem::GemmRowSoftmaxTileShape::kGemmK;
    constexpr index_t kMaxK = Problem::GemmRowSoftmaxTileShape::kMaxK;

    constexpr index_t K2 = min(kMPerBlock * kKPerBlock / kBlockSize,
                               static_cast<index_t>(8 / sizeof(A0DataType)));
    constexpr index_t K1 = kKPerBlock / K2;
    constexpr index_t K0 = kMaxK / kKPerBlock;
    constexpr index_t M2 = get_warp_size() / K1;
    constexpr index_t M1 = kBlockSize / get_warp_size();
    constexpr index_t M0 = kMPerBlock / (M2 * M1);

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>, tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
            tuple<sequence<1>, sequence<1, 2>>,
            tuple<sequence<1>, sequence<2, 1>>, sequence<1, 2, 2>,
            sequence<0, 0, 2>>{});
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeA0LdsBlockDescriptor() {
    using A0DataType = remove_cvref_t<typename Problem::InOutDataType>;

    constexpr index_t kMPerBlock = Problem::GemmRowSoftmaxTileShape::kGemmM;
    constexpr index_t kKPerBlock = Problem::GemmRowSoftmaxTileShape::kGemmK;
    constexpr index_t kMaxK = Problem::GemmRowSoftmaxTileShape::kMaxK;
    constexpr index_t kKPack = 8 / sizeof(A0DataType);

    constexpr auto a0_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kMaxK / kKPerBlock>{}, number<kKPerBlock / kKPack>{},
                   number<kMPerBlock>{}, number<kKPack>{}),
        make_tuple(number<(kMPerBlock + 1) * kKPerBlock>{},
                   number<(kMPerBlock + 1) * kKPack>{}, number<kKPack>{},
                   number<1>{}),
        number<4>{}, number<1>{});

    constexpr auto a0_lds_block_desc = transform_tensor_descriptor(
        a0_lds_block_desc_0,
        make_tuple(make_pass_through_transform(number<kMPerBlock>{}),
                   make_merge_transform(make_tuple(
                       number<kMaxK / kKPerBlock>{},
                       number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<2>{}, sequence<0, 1, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return a0_lds_block_desc;
  }

  template <typename Problem, typename BlockGemm>
  CK_TILE_HOST_DEVICE static constexpr auto MakeMaskDramTileDistribution() {
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t kMPerBlock = Problem::GemmRowSoftmaxTileShape::kGemmM;
    constexpr index_t kNPerBlock = Problem::GemmRowSoftmaxTileShape::kGemmN;

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

    constexpr auto mask_block_outer_dstr_encoding =
        tile_distribution_encoding<sequence<M0, M1>, tuple<sequence<N0, N1>>,
                                   tuple<sequence<0, 1>>, tuple<sequence<1, 1>>,
                                   sequence<1>, sequence<0>>{};

    constexpr auto mask_block_dstr_encoding =
        detail::make_embed_tile_distribution_encoding(
            mask_block_outer_dstr_encoding, mask_warp_dstr_encoding);

    constexpr auto mask_block_dstr =
        make_static_tile_distribution(mask_block_dstr_encoding);

    return mask_block_dstr;
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetA0B0TBlockGemm() {
    using BlockGemmProblem = BlockGemmPipelineProblem<
        typename Problem::InOutDataType, typename Problem::InOutDataType,
        typename Problem::GemmAccDataType, Problem::kGemm0BlockSize,
        TileGemmShape<Problem::GemmRowSoftmaxTileShape::kGemmM,
                      Problem::GemmRowSoftmaxTileShape::kGemmN,
                      Problem::GemmRowSoftmaxTileShape::kGemmK>>;

    constexpr auto warp_gemm = []() {
      if constexpr (std::is_same_v<typename Problem::InOutDataType, fp16_t> &&
                    std::is_same_v<typename Problem::GemmAccDataType, float>) {
        return WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution{};
      } else if constexpr (std::is_same_v<typename Problem::InOutDataType,
                                          bf16_t> &&
                           std::is_same_v<typename Problem::GemmAccDataType,
                                          float>) {
        return WarpGemmMfmaBf16Bf16F32M32N32K8TransposedCDistribution{};
      }
    }();

    using BlockGemmPolicy = BlockGemmASmemBRegCRegV1CustomPolicy<
        typename Problem::InOutDataType, typename Problem::InOutDataType,
        typename Problem::GemmAccDataType,
        typename Problem::GemmRowSoftmaxTileShape::GemmBlockWarps,
        decltype(warp_gemm)>;

    return BlockGemmASmemBRegCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
  }

  template <typename Problem, typename BlockGemm>
  CK_TILE_HOST_DEVICE static constexpr auto MakeB1TDramTileDistribution() {
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t kNPerBlock = Problem::GemmTileShape::kGemmN;
    constexpr index_t kMaxK = Problem::GemmTileShape::kMaxK;

    // K2 is equal to Impl::kABKPerLane * kKIter
    constexpr index_t K2 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
    constexpr index_t K0 = kMaxK / (K1 * K2);

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

  template <typename Problem, typename BlockGemm>
  CK_TILE_HOST_DEVICE static constexpr auto MakeA1DramTileDistribution() {
    using A1DataType = remove_cvref_t<typename Problem::InOutDataType>;

    constexpr index_t kBlockSize = Problem::kGemm1BlockSize;
    constexpr index_t kMPerBlock = Problem::GemmTileShape::kGemmM;
    constexpr index_t kKPerBlock = Problem::GemmTileShape::kGemmK;
    constexpr index_t kMaxK = Problem::GemmTileShape::kMaxK;

    static_assert(kKPerBlock <= kMaxK, "kGemmK must not be bigger than kMaxK!");

    constexpr index_t K2 = min(kMPerBlock * kKPerBlock / kBlockSize,
                               static_cast<index_t>(8 / sizeof(A1DataType)));
    constexpr index_t K1 = kKPerBlock / K2;
    constexpr index_t K0 = kMaxK / kKPerBlock;
    constexpr index_t M2 = get_warp_size() / K1;
    constexpr index_t M1 = kBlockSize / get_warp_size();
    constexpr index_t M0 = kMPerBlock / (M2 * M1);

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>, tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
            tuple<sequence<1>, sequence<1, 2>>,
            tuple<sequence<1>, sequence<2, 1>>, sequence<1, 2, 2>,
            sequence<0, 0, 2>>{});
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeA1LdsBlockDescriptor() {
    using A1DataType = remove_cvref_t<typename Problem::InOutDataType>;

    constexpr index_t kMPerBlock = Problem::GemmTileShape::kGemmM;
    constexpr index_t kKPerBlock = Problem::GemmTileShape::kGemmK;
    constexpr index_t kMaxK = Problem::GemmTileShape::kMaxK;
    constexpr index_t kKPack = 8 / sizeof(A1DataType);

    constexpr auto a1_lds_block_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kMaxK / kKPerBlock>{}, number<kKPerBlock / kKPack>{},
                   number<kMPerBlock>{}, number<kKPack>{}),
        make_tuple(number<(kMPerBlock + 1) * kKPerBlock>{},
                   number<(kMPerBlock + 1) * kKPack>{}, number<kKPack>{},
                   number<1>{}),
        number<4>{}, number<1>{});

    constexpr auto a1_lds_block_desc = transform_tensor_descriptor(
        a1_lds_block_desc_0,
        make_tuple(make_pass_through_transform(number<kMPerBlock>{}),
                   make_merge_transform(make_tuple(
                       number<kMaxK / kKPerBlock>{},
                       number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<2>{}, sequence<0, 1, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return a1_lds_block_desc;
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetA1B1TBlockGemm() {
    using BlockGemmProblem = BlockGemmPipelineProblem<
        typename Problem::InOutDataType, typename Problem::InOutDataType,
        typename Problem::GemmAccDataType, Problem::kGemm1BlockSize,
        TileGemmShape<Problem::GemmTileShape::kGemmM,
                      Problem::GemmTileShape::kGemmN,
                      Problem::GemmTileShape::kGemmK>>;

    constexpr auto warp_gemm = []() {
      if constexpr (std::is_same_v<typename Problem::InOutDataType, fp16_t> &&
                    std::is_same_v<typename Problem::GemmAccDataType, float>) {
        return WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution{};
      } else if constexpr (std::is_same_v<typename Problem::InOutDataType,
                                          bf16_t> &&
                           std::is_same_v<typename Problem::GemmAccDataType,
                                          float>) {
        return WarpGemmMfmaBf16Bf16F32M32N32K16TransposedCDistribution{};
      }
    }();

    using BlockGemmPolicy = BlockGemmASmemBRegCRegV1CustomPolicy<
        typename Problem::InOutDataType, typename Problem::InOutDataType,
        typename Problem::GemmAccDataType,
        typename Problem::GemmTileShape::GemmBlockWarps, decltype(warp_gemm)>;

    return BlockGemmASmemBRegCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
  }
};

}  // namespace ck_tile
