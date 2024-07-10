// // SPDX-License-Identifier: MIT
// // Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

// #pragma once

// #include "ck_tile/core.hpp"
// #include "ck_tile/ops/common/tensor_layout.hpp"
// #include "ck_tile/ops/gemm/pipeline/block_gemm_pipeline_problem.hpp"
// #include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
// #include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
// #include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp"
// #include "ck_tile/core/tensor/tensor_adaptor.hpp"
// #include "ck_tile/core/algorithm/coordinate_transform.hpp"

// // Overloaded function to provide a default RightShift
// template <typename LowLengths>
// CK_TILE_HOST_DEVICE constexpr auto make_xor_transform(const LowLengths& low_lengths)
// {
//     int default_right_shift = 0; // Default value for RightShift
//     return ck_tile::xor_t<LowLengths, int>{low_lengths, default_right_shift};
// }

// namespace ck_tile {
// // This pipeline is qwkv all located in LDS.
// struct BlockGemmLnAttnPipelineQWKVldsCustomPolicy
// {
//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetQWBlockGemm()
//     {
//         using BlockGemmProblem =
//             BlockGemmPipelineProblem<typename Problem::QDataType,
//                                      typename Problem::WDataType,
//                                      typename Problem::S0accDataType,
//                                      Problem::kBlockSize,
//                                      TileGemmShape<Problem::BlockGemmLnAttnShape::kM0,
//                                                    Problem::BlockGemmLnAttnShape::kN0,
//                                                    Problem::BlockGemmLnAttnShape::kK0>>;

//         auto warp_gemm =
//             WarpGemmMfmaDispatcher<typename Problem::QDataType,
//                                    typename Problem::WDataType,
//                                    typename Problem::S0accDataType,
//                                    Problem::BlockGemmLnAttnShape::Gemm0WarpTile::at(number<0>{}),
//                                    Problem::BlockGemmLnAttnShape::Gemm0WarpTile::at(number<1>{}),
//                                    Problem::BlockGemmLnAttnShape::Gemm0WarpTile::at(number<2>{}),
//                                    true>{};

//         using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<
//             typename Problem::QDataType,
//             typename Problem::WDataType,
//             typename Problem::S0accDataType,
//             typename Problem::BlockGemmLnAttnShape::Gemm0BlockWarps,
//             decltype(warp_gemm)>;

//         return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetRKBlockGemm()
//     {
//         using BlockGemmProblem =
//             BlockGemmPipelineProblem<typename Problem::RDataType,
//                                      typename Problem::KDataType,
//                                      typename Problem::S1accDataType,
//                                      Problem::kBlockSize,
//                                      TileGemmShape<Problem::BlockGemmLnAttnShape::kM0,
//                                                    Problem::BlockGemmLnAttnShape::kN1,
//                                                    Problem::BlockGemmLnAttnShape::kK1>>;

//         auto warp_gemm =
//             WarpGemmMfmaDispatcher<typename Problem::RDataType,
//                                    typename Problem::KDataType,
//                                    typename Problem::S1accDataType,
//                                    Problem::BlockGemmLnAttnShape::Gemm1WarpTile::at(number<0>{}),
//                                    Problem::BlockGemmLnAttnShape::Gemm1WarpTile::at(number<1>{}),
//                                    Problem::BlockGemmLnAttnShape::Gemm1WarpTile::at(number<2>{}),
//                                    true>{};

//         using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
//             typename Problem::RDataType,
//             typename Problem::KDataType,
//             typename Problem::S1accDataType,
//             typename Problem::BlockGemmLnAttnShape::Gemm1BlockWarps,
//             decltype(warp_gemm)>;

//         return BlockGemmARegBSmemCRegV2<BlockGemmProblem, BlockGemmPolicy>{};
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetPVBlockGemm()
//     {
//         using BlockGemmProblem =
//             BlockGemmPipelineProblem<typename Problem::PDataType,
//                                      typename Problem::VDataType,
//                                      typename Problem::OaccDataType,
//                                      Problem::kBlockSize,
//                                      TileGemmShape<Problem::BlockGemmLnAttnShape::kM0,
//                                                    Problem::BlockGemmLnAttnShape::kN2,
//                                                    Problem::BlockGemmLnAttnShape::kK2>>;

//         auto warp_gemm =
//             WarpGemmMfmaDispatcher<typename Problem::PDataType,
//                                    typename Problem::VDataType,
//                                    typename Problem::OaccDataType,
//                                    Problem::BlockGemmLnAttnShape::Gemm2WarpTile::at(number<0>{}),
//                                    Problem::BlockGemmLnAttnShape::Gemm2WarpTile::at(number<1>{}),
//                                    Problem::BlockGemmLnAttnShape::Gemm2WarpTile::at(number<2>{}),
//                                    true>{};

//         using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

//         using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
//             typename Problem::PDataType,
//             typename Problem::VDataType,
//             typename Problem::OaccDataType,
//             typename Problem::BlockGemmLnAttnShape::Gemm2BlockWarps,
//             WarpGemm>;

//         return BlockGemmARegBSmemCRegV2<BlockGemmProblem, BlockGemmPolicy>{};
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
//     {
//         using QDataType = remove_cvref_t<typename Problem::QDataType>;
//         return 16 / sizeof(QDataType);
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentW()
//     {
//         constexpr index_t kBlockSize   = Problem::kBlockSize;
//         constexpr index_t kNPerBlock   = Problem::BlockGemmLnAttnShape::kN0;
//         constexpr index_t kKPerBlock   = Problem::BlockGemmLnAttnShape::kK0;
//         constexpr index_t kTotalPixels = kNPerBlock * kKPerBlock / kBlockSize;

//         using WDataType               = remove_cvref_t<typename Problem::WDataType>;
//         constexpr index_t kMaxVecLoad = 16 / sizeof(WDataType);

//         return (kTotalPixels > kMaxVecLoad) ? kMaxVecLoad : kTotalPixels;
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentBias()
//     {
//         // WG::WarpGemmAttribute::Impl::kCM1PerLane;
//         return 4;
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentGamma()
//     {
//         // WG::WarpGemmAttribute::Impl::kCM1PerLane;
//         return 4;
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentBeta()
//     {
//         // WG::WarpGemmAttribute::Impl::kCM1PerLane;
//         return 4;
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
//     {
//         constexpr index_t kBlockSize   = Problem::kBlockSize;
//         constexpr index_t kNPerBlock   = Problem::BlockGemmLnAttnShape::kN1;
//         constexpr index_t kKPerBlock   = Problem::BlockGemmLnAttnShape::kK1;
//         constexpr index_t kTotalPixels = kNPerBlock * kKPerBlock / kBlockSize;

//         using KDataType               = remove_cvref_t<typename Problem::KDataType>;
//         constexpr index_t kMaxVecLoad = 16 / sizeof(KDataType);

//         return (kTotalPixels > kMaxVecLoad) ? kMaxVecLoad : kTotalPixels;
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentKeyMask()
//     {
//         // WG::WarpGemmAttribute::Impl::kCM1PerLane;
//         return 4;
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
//     {
//         using VDataType               = remove_cvref_t<typename Problem::VDataType>;
//         constexpr index_t kBlockSize  = Problem::kBlockSize;
//         constexpr index_t kNPerBlock  = Problem::BlockGemmLnAttnShape::kN2;
//         constexpr index_t kKPerBlock  = Problem::BlockGemmLnAttnShape::kK2;
//         constexpr index_t kMaxVecLoad = 16 / sizeof(VDataType);
//         constexpr index_t kMinVecLoad = 4 / sizeof(VDataType);

//         constexpr index_t kTotalPixels = kNPerBlock * kKPerBlock / kBlockSize;

//         constexpr index_t kVecLoad = ((kTotalPixels / kMaxVecLoad) >= kMinVecLoad)
//                                          ? kMaxVecLoad
//                                          : (kTotalPixels / kMinVecLoad);

//         return kVecLoad;
//     }

//     template <typename Problem>
//     __host__ __device__ static constexpr auto GetTransposedAlignmentV()
//     {
//         constexpr index_t kBlockSize   = Problem::kBlockSize;
//         constexpr index_t kNPerBlock   = Problem::BlockGemmLnAttnShape::kN2;
//         constexpr index_t kKPerBlock   = Problem::BlockGemmLnAttnShape::kK2;
//         constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

//         return total_pixels / GetAlignmentV<Problem>();
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
//     {
//         using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
//         constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
//         using WG              = remove_cvref_t<decltype(config.template at<0>())>;
//         using CWarpDstr       = typename WG::CWarpDstr;
//         constexpr auto vec =
//             CWarpDstr{}.get_ys_to_d_descriptor().get_lengths().at(number<CWarpDstr::NDimY - 1>{});
//         return vec;
//     }

//     // HBM descriptors
//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
//     {
//         constexpr index_t kBlockSize = Problem::kBlockSize;

//         constexpr index_t kMPerBlock = Problem::BlockGemmLnAttnShape::kM0;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK0;

//         constexpr index_t K1 = GetAlignmentQ<Problem>();
//         constexpr index_t K0 = kKPerBlock / K1;
//         constexpr index_t M2 = get_warp_size() / K0;
//         constexpr index_t M1 = kBlockSize / get_warp_size();
//         constexpr index_t M0 = kMPerBlock / (M2 * M1);

//         return make_static_tile_distribution(
//             tile_distribution_encoding<sequence<1>,
//                                        tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
//                                        tuple<sequence<1>, sequence<1, 2>>,
//                                        tuple<sequence<1>, sequence<2, 0>>,
//                                        sequence<1, 2>,
//                                        sequence<0, 1>>{});
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeWDramTileDistribution()
//     {
//         constexpr index_t kBlockSize = Problem::kBlockSize;

//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN0;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK0;

//         constexpr index_t K1 = GetAlignmentW<Problem>();
//         constexpr index_t K0 = kKPerBlock / K1;
//         constexpr index_t N2 = get_warp_size() / K0;
//         constexpr index_t N1 = kBlockSize / get_warp_size();
//         constexpr index_t N0 = kNPerBlock / (N2 * N1);

//         return make_static_tile_distribution(
//             tile_distribution_encoding<sequence<1>,
//                                        tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
//                                        tuple<sequence<1>, sequence<1, 2>>,
//                                        tuple<sequence<1>, sequence<2, 0>>,
//                                        sequence<1, 2>,
//                                        sequence<0, 1>>{});
//     }

//     template <typename Problem, typename BlockGemm>
//     __host__ __device__ static constexpr auto MakeBiasGammaBetaDramTileDistribution()
//     {
//         constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
//         using WG                = remove_cvref_t<decltype(config.template at<0>())>;
//         constexpr index_t MWarp = config.template at<1>();
//         constexpr index_t NWarp = config.template at<2>();
//         static_assert(NWarp == 1, "Warp in N direction should be 1 in 1st Gemm");

//         // Gemm0 should be transposed
//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN0;

//         constexpr index_t M1 = WG::WarpGemmAttribute::Impl::kCNLane;
//         constexpr index_t M0 = MWarp;

//         constexpr index_t N4 = WG::WarpGemmAttribute::Impl::kCM1PerLane;
//         constexpr index_t N3 = WG::WarpGemmAttribute::Impl::kCMLane;
//         constexpr index_t N2 = WG::WarpGemmAttribute::Impl::kCM0PerLane;
//         constexpr index_t N1 = NWarp;
//         constexpr index_t N0 = kNPerBlock / (N1 * WG::WarpGemmAttribute::Impl::kM);

//         return make_static_tile_distribution(
//             tile_distribution_encoding<sequence<M0, M1>,
//                                        tuple<sequence<N0, N1, N2, N3, N4>>,
//                                        tuple<sequence<1, 0>, sequence<1, 0>>,
//                                        tuple<sequence<1, 0>, sequence<3, 1>>,
//                                        sequence<1, 1, 1>,
//                                        sequence<0, 2, 4>>{});
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
//     {
//         constexpr index_t kBlockSize = Problem::kBlockSize;

//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN1;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK1;

//         constexpr index_t K1 = GetAlignmentK<Problem>();
//         constexpr index_t K0 = kKPerBlock / K1;
//         constexpr index_t N2 = get_warp_size() / K0;
//         constexpr index_t N1 = kBlockSize / get_warp_size();
//         constexpr index_t N0 = kNPerBlock / (N2 * N1);

//         return make_static_tile_distribution(
//             tile_distribution_encoding<sequence<1>,
//                                        tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
//                                        tuple<sequence<1>, sequence<1, 2>>,
//                                        tuple<sequence<1>, sequence<2, 0>>,
//                                        sequence<1, 2>,
//                                        sequence<0, 1>>{});
//     }

//     template <typename Problem, typename BlockGemm>
//     __host__ __device__ static constexpr auto MakeKeyMaskDramTileDistribution()
//     {
//         constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
//         using WG                = remove_cvref_t<decltype(config.template at<0>())>;
//         constexpr index_t MWarp = config.template at<1>();
//         constexpr index_t NWarp = config.template at<2>();
//         static_assert(NWarp == 1, "Warp in N direction should be 1 in 1st Gemm");

//         // Gemm0 should be transposed
//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN1;

//         constexpr index_t M1 = WG::WarpGemmAttribute::Impl::kCNLane;
//         constexpr index_t M0 = MWarp;

//         constexpr index_t N4 = WG::WarpGemmAttribute::Impl::kCM1PerLane;
//         constexpr index_t N3 = WG::WarpGemmAttribute::Impl::kCMLane;
//         constexpr index_t N2 = WG::WarpGemmAttribute::Impl::kCM0PerLane;
//         constexpr index_t N1 = NWarp;
//         constexpr index_t N0 = kNPerBlock / (N1 * WG::WarpGemmAttribute::Impl::kM);

//         return make_static_tile_distribution(
//             tile_distribution_encoding<sequence<M0, M1>,
//                                        tuple<sequence<N0, N1, N2, N3, N4>>,
//                                        tuple<sequence<1, 0>, sequence<1, 0>>,
//                                        tuple<sequence<1, 0>, sequence<3, 1>>,
//                                        sequence<1, 1, 1>,
//                                        sequence<0, 2, 4>>{});
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
//     {
//         // N Major
//         constexpr index_t kBlockSize = Problem::kBlockSize;

//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN2;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK2;

//         constexpr index_t N1 = GetAlignmentV<Problem>();
//         constexpr index_t N0 = kNPerBlock / N1;
//         constexpr index_t K1 = get_warp_size() / N0;
//         constexpr index_t K0 = kBlockSize / get_warp_size();
//         constexpr index_t K2 = kKPerBlock / (K1 * K0);

//         return make_static_tile_distribution(
//             tile_distribution_encoding<sequence<1>,
//                                        tuple<sequence<K0, K1, K2>, sequence<N0, N1>>,
//                                        tuple<sequence<1>, sequence<1, 2>>,
//                                        tuple<sequence<0>, sequence<1, 0>>,
//                                        sequence<1, 2>,
//                                        sequence<2, 1>>{});
//     }

//     // LDS descriptors
//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
//     {
//         return GetAlignmentQ<Problem>();
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackW()
//     {
//         return GetAlignmentW<Problem>();
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
//     {
//         return GetAlignmentK<Problem>();
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
//     {
//         return GetTransposedAlignmentV<Problem>();
//     }

//     template <typename DataType, index_t MNPerBlock, index_t KPerBlock, index_t KPack>
//     __host__ __device__ static constexpr auto MakeXLdsBlockDescriptor()
//     {
//         constexpr auto DataTypeSize = sizeof(DataType);
//         constexpr auto MNLdsLayer =
//             (32 * 4 / KPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerBlock / DataTypeSize);

//         constexpr auto x_lds_block_desc_0 = ck_tile::make_naive_tensor_descriptor(
//             ck_tile::make_tuple(number<KPerBlock / KPack * MNLdsLayer>{},
//                        number<MNPerBlock / MNLdsLayer>{},
//                        number<KPack>{}),
//             ck_tile::make_tuple(number<KPack>{}, number<KPerBlock * MNLdsLayer>{}, number<1>{}),
//             number<KPack>{},
//             number<1>{});

//         constexpr auto x_lds_block_desc_permuted = transform_tensor_descriptor(
//             x_lds_block_desc_0,
//             ck_tile::make_tuple(ck_tile::make_xor_transform(ck_tile::make_tuple(number<MNPerBlock / MNLdsLayer>{},
//                                                      number<KPerBlock / KPack * MNLdsLayer>{}), 0),
//                        make_pass_through_transform(number<KPack>{})),
//             ck_tile::make_tuple(sequence<1, 0>{}, sequence<2>{}),
//             ck_tile::make_tuple(sequence<1, 0>{}, sequence<2>{}));

//         constexpr auto x_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
//             x_lds_block_desc_permuted,
//             ck_tile::make_tuple(make_unmerge_transform(
//                            ck_tile::make_tuple(number<KPerBlock / KPack>{}, number<MNLdsLayer>{})),
//                        make_pass_through_transform(number<MNPerBlock / MNLdsLayer>{}),
//                        make_pass_through_transform(number<KPack>{})),
//             ck_tile::make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
//             ck_tile::make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

//         constexpr auto x_lds_block_desc = transform_tensor_descriptor(
//             x_lds_block_desc_xk0_mnldslayer_mn_xk1,
//             ck_tile::make_tuple(make_merge_transform_v3_division_mod(
//                            ck_tile::make_tuple(number<MNPerBlock / MNLdsLayer>{}, number<MNLdsLayer>{})),
//                        make_merge_transform_v3_division_mod(
//                            ck_tile::make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
//             ck_tile::make_tuple(sequence<1, 2>{}, sequence<0, 3>{}),
//             ck_tile::make_tuple(sequence<0>{}, sequence<1>{}));

//         return x_lds_block_desc;
//     }

//     template <typename Problem,
//               typename DataType,
//               index_t MNPerBlock,
//               index_t KPerBlock,
//               index_t KPack,
//               index_t KPackT>
//     __host__ __device__ static constexpr auto MakeXTLdsBlockDescriptor()
//     {
//         // kfold and mpair dimension is not always required.
//         // more dimension in merge_transform increase the difficulty of generating immarg offset
//         // for compiler.
//         constexpr auto DataTypeSize = sizeof(DataType);

//         constexpr auto MNPerXDL   = Problem::BlockGemmLnAttnShape::Gemm0WarpTile::at(number<0>{});
//         constexpr auto kBlockSize = Problem::kBlockSize;

//         constexpr auto MN0 = MNPerBlock / KPack;
//         constexpr auto MN1 = KPack;

//         constexpr auto KThreadWrite     = kBlockSize / MN0;
//         constexpr auto K0number         = KPerBlock / KPackT;
//         constexpr auto K0PerThreadWrite = K0number / KThreadWrite;
//         constexpr auto KThreadRead      = get_warp_size() / MNPerXDL;
//         constexpr auto K0PerThreadRead  = K0number / KThreadRead;

//         constexpr auto kfold =
//             (KPackT * MN0 * DataTypeSize > 128) ? 1 : 128 / (KPackT * MN0 * DataTypeSize);
//         constexpr auto KThreadReadPerm =
//             (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
//                 ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
//                 : KThreadRead;

//         // 1<=mnpair<=n0
//         constexpr auto mnpair = (KPackT * MNPerXDL * DataTypeSize > 128)
//                                     ? 1
//                                     : ((128 / (KPackT * MNPerXDL * DataTypeSize)) > MN0
//                                            ? MN0
//                                            : 128 / (KPackT * MNPerXDL * DataTypeSize));

//         constexpr auto xt_lds_block_desc_raw = ck_tile::make_naive_tensor_descriptor(
//             ck_tile::make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
//                        number<K0PerThreadWrite>{},
//                        number<KThreadReadPerm * MN1>{},
//                        number<kfold * MN0 / mnpair>{},
//                        number<mnpair>{},
//                        KPackT),
//             ck_tile::make_tuple(number<KPackT * kfold * MN0 * KThreadReadPerm * MN1 * K0PerThreadWrite>{},
//                        number<KPackT * kfold * MN0 * KThreadReadPerm * MN1>{},
//                        number<KPackT * kfold * MN0>{},
//                        number<KPackT * mnpair>{},
//                        number<KPackT>{},
//                        number<1>{}),
//             number<KPackT>{},
//             number<1>{});

//         constexpr auto xt_lds_block_desc_permuted = transform_tensor_descriptor(
//             xt_lds_block_desc_raw,
//             ck_tile::make_tuple(
//                 make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
//                 make_pass_through_transform(number<K0PerThreadWrite>{}),
//                 ck_tile::make_xor_transform(
//                     ck_tile::make_tuple(number<KThreadReadPerm * MN1>{}, number<kfold * MN0 / mnpair>{}), 0),
//                 make_pass_through_transform(number<mnpair>{}),
//                 make_pass_through_transform(KPackT)),
//             ck_tile::make_tuple(
//                 sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
//             ck_tile::make_tuple(
//                 sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

//         constexpr auto xt_lds_block_desc_unmerged = transform_tensor_descriptor(
//             xt_lds_block_desc_permuted,
//             ck_tile::make_tuple(
//                 make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
//                 make_pass_through_transform(number<K0PerThreadWrite>{}),
//                 make_unmerge_transform(ck_tile::make_tuple(number<KThreadReadPerm>{}, number<MN1>{})),
//                 make_unmerge_transform(ck_tile::make_tuple(number<kfold>{}, number<MN0 / mnpair>{})),
//                 make_pass_through_transform(number<mnpair>{}),
//                 make_pass_through_transform(KPackT)),
//             ck_tile::make_tuple(sequence<0>{},
//                        sequence<1>{},
//                        sequence<2>{},
//                        sequence<3>{},
//                        sequence<4>{},
//                        sequence<5>{}),
//             ck_tile::make_tuple(sequence<1>{},
//                        sequence<2>{},
//                        sequence<0, 3>{},
//                        sequence<4, 5>{},
//                        sequence<6>{},
//                        sequence<7>{}));

//         constexpr auto xt_lds_block_desc = transform_tensor_descriptor(
//             xt_lds_block_desc_unmerged,
//             ck_tile::make_tuple(make_merge_transform_v3_division_mod(
//                            ck_tile::make_tuple(number<KThreadReadPerm>{},
//                                       number<KThreadWrite / kfold / KThreadReadPerm>{},
//                                       number<kfold>{},
//                                       number<K0PerThreadWrite>{},
//                                       number<KPackT>{})),
//                        make_merge_transform_v3_division_mod(
//                            ck_tile::make_tuple(number<MN0 / mnpair>{}, number<mnpair>{}, number<MN1>{}))),
//             ck_tile::make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
//             ck_tile::make_tuple(sequence<0>{}, sequence<1>{}));

//         return xt_lds_block_desc;
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
//     {
//         using QDataType = remove_cvref_t<typename Problem::QDataType>;

//         constexpr index_t kMPerBlock = Problem::BlockGemmLnAttnShape::kM0;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK0;
//         constexpr index_t kKPack     = GetSmemKPackQ<Problem>();

//         return MakeXLdsBlockDescriptor<QDataType, kMPerBlock, kKPerBlock, kKPack>();
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeWLdsBlockDescriptor()
//     {
//         using WDataType = remove_cvref_t<typename Problem::WDataType>;

//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN0;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK0;
//         constexpr index_t kKPack     = GetSmemKPackW<Problem>();

//         return MakeXLdsBlockDescriptor<WDataType, kNPerBlock, kKPerBlock, kKPack>();
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
//     {
//         using KDataType = remove_cvref_t<typename Problem::KDataType>;

//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN1;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK1;
//         constexpr index_t kKPack     = GetSmemKPackK<Problem>();

//         return MakeXLdsBlockDescriptor<KDataType, kNPerBlock, kKPerBlock, kKPack>();
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
//     {
//         using VDataType = remove_cvref_t<typename Problem::VDataType>;

//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN2;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK2;
//         constexpr index_t kKPack     = GetAlignmentV<Problem>();
//         constexpr index_t kKPackT    = GetSmemKPackV<Problem>();
//         // static_assert(kKPack==4 && kKPackT==2, "Error: VLdsWrite");

//         return MakeXTLdsBlockDescriptor<Problem,
//                                         VDataType,
//                                         kNPerBlock,
//                                         kKPerBlock,
//                                         kKPack,
//                                         kKPackT>();
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsReadBlockDescriptor()
//     {
//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN2;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK2;

//         auto v_lds_block_desc = MakeVLdsWriteBlockDescriptor<Problem>();

//         return transform_tensor_descriptor(
//             v_lds_block_desc,
//             ck_tile::make_tuple(make_pass_through_transform(number<kNPerBlock>{}),
//                        make_pass_through_transform(number<kKPerBlock>{})),
//             ck_tile::make_tuple(sequence<1>{}, sequence<0>{}),
//             ck_tile::make_tuple(sequence<0>{}, sequence<1>{}));
//     }

//     template <typename Problem>
//     __host__ __device__ static constexpr index_t GetSmemSizeQ()
//     {
//         constexpr index_t smem_size_q = sizeof(typename Problem::QDataType) *
//                                         MakeQLdsBlockDescriptor<Problem>().get_element_space_size();
//         return smem_size_q;
//     }

//     template <typename Problem>
//     __host__ __device__ static constexpr index_t GetSmemSizeW()
//     {
//         constexpr index_t smem_size_w = sizeof(typename Problem::WDataType) *
//                                         MakeWLdsBlockDescriptor<Problem>().get_element_space_size();
//         return smem_size_w;
//     }

//     template <typename Problem>
//     __host__ __device__ static constexpr index_t GetSmemSizeK()
//     {
//         constexpr index_t smem_size_k = sizeof(typename Problem::KDataType) *
//                                         MakeKLdsBlockDescriptor<Problem>().get_element_space_size();
//         return smem_size_k;
//     }

//     template <typename Problem>
//     __host__ __device__ static constexpr index_t GetSmemSizeV()
//     {
//         constexpr index_t smem_size_v =
//             sizeof(typename Problem::VDataType) *
//             MakeVLdsWriteBlockDescriptor<Problem>().get_element_space_size();
//         return smem_size_v;
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
//     {
//         constexpr index_t smem_size_q = GetSmemSizeQ<Problem>();
//         constexpr index_t smem_size_w = GetSmemSizeW<Problem>();
//         constexpr index_t smem_size_k = GetSmemSizeK<Problem>();
//         constexpr index_t smem_size_v = GetSmemSizeV<Problem>();

//         constexpr index_t smem_size_stage1 = smem_size_q + smem_size_w;
//         constexpr index_t smem_size_stage2 = smem_size_k + smem_size_v;

//         return ck_tile::max(smem_size_stage1, smem_size_stage2);
//     }

//     // Reg descriptors
//     template <typename Problem, typename BlockGemm>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeRRegBlockDescriptor()
//     {
//         constexpr index_t kMPerBlock = Problem::BlockGemmLnAttnShape::kM0;
//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN0;

//         constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();

//         using WG = remove_cvref_t<decltype(config.template at<0>())>;

//         constexpr index_t MWarp = config.template at<1>();
//         constexpr index_t NWarp = config.template at<2>();

//         constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
//         constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);

//         constexpr auto r_block_outer_dstr_encoding = tile_distribution_encoding<
//             sequence<>,
//             tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
//             tuple<sequence<1, 2>>,
//             tuple<sequence<1, 1>>,
//             sequence<1, 2>,
//             sequence<0, 0>>{};

//         constexpr auto r_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
//             r_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

//         constexpr auto r_block_dstr = make_static_tile_distribution(r_block_dstr_encode);

//         return r_block_dstr;
//     }

//     template <typename Problem>
//     CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledVRegBlockDescriptor()
//     {
//         // N Major
//         constexpr index_t kBlockSize = Problem::kBlockSize;

//         constexpr index_t kNPerBlock = Problem::BlockGemmLnAttnShape::kN2;
//         constexpr index_t kKPerBlock = Problem::BlockGemmLnAttnShape::kK2;

//         constexpr index_t N1 = GetAlignmentV<Problem>();
//         constexpr index_t N0 = kNPerBlock / N1;
//         constexpr index_t K1 = get_warp_size() / N0;
//         constexpr index_t K0 = kBlockSize / get_warp_size();
//         constexpr index_t K2 = kKPerBlock / (K1 * K0);

//         return make_static_tile_distribution(
//             tile_distribution_encoding<sequence<1>,
//                                        tuple<sequence<K0, K1, K2>, sequence<N0, N1>>,
//                                        tuple<sequence<1>, sequence<1, 2>>,
//                                        tuple<sequence<0>, sequence<1, 0>>,
//                                        sequence<2, 1>,
//                                        sequence<1, 2>>{});
//     }

//     template <typename Problem, typename BlockGemm>
//     CK_TILE_HOST_DEVICE static constexpr auto GetLayerNormIntraLaneReduceCount(index_t NLength)
//     {
//         constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
//         using WG                = remove_cvref_t<decltype(config.template at<0>())>;
//         constexpr index_t NWarp = config.template at<2>();
//         static_assert(NWarp == 1, "Warp in N direction should be 1 in 1st Gemm");

//         // Transposed
//         // NPerBlock composed by NIter -> NWarp(=1) -> WG::kN
//         // WG::kN composed by  kCM0PerLane -> kCMLane -> kCM1PerLane
//         constexpr index_t N0 = WG::WarpGemmAttribute::Impl::kM;
//         // constexpr index_t N1 = WG::WarpGemmAttribute::Impl::kCM0PerLane;
//         constexpr index_t N2 = WG::WarpGemmAttribute::Impl::kCMLane;
//         constexpr index_t N3 = WG::WarpGemmAttribute::Impl::kCM1PerLane;

//         // Count = iNIter * N1 * N3 + iN1* N3 + iN3
//         index_t iNLane = (get_lane_id() % get_warp_size()) / WG::WarpGemmAttribute::Impl::kN;
//         index_t iN0    = NLength / N0;
//         index_t iN1    = (NLength % N0) / (N2 * N3);
//         index_t iN2    = (NLength % (N2 * N3)) / N3;
//         index_t iN3    = iNLane < iN2 ? N3 : iNLane = iN2 ? NLength % N3 : 0;

//         return iN0 * N0 + iN1 * N3 + iN3;
//     }
// };

// } // namespace ck_tile
