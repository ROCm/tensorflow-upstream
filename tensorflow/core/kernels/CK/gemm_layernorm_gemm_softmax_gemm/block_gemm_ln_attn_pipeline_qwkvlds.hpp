// // SPDX-License-Identifier: MIT
// // Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

// #pragma once

// #include "ck_tile/core.hpp"
// #include "block_gemm_ln_attn_pipeline_pwkvlds_custom_policy.hpp"
// #include "ck_tile/ops/reduce/block/block_reduce.hpp"
// #include "ck_tile/ops/welford/thread/thread_welford.hpp"
// #include "ck_tile/ops/welford/warp/warp_welford.hpp"

// namespace ck_tile {

// using BlockGemmLnAttnPipelineQWKVldsDefaultPolicy = BlockGemmLnAttnPipelineQWKVldsCustomPolicy;

// template <typename Problem_, typename Policy_ = BlockGemmLnAttnPipelineQWKVldsDefaultPolicy>
// struct BlockGemmLnAttnPipelineQWKVlds
// {
//     using Problem             = remove_cvref_t<Problem_>;
//     using Policy              = remove_cvref_t<Policy_>;
//     using QDataType           = remove_cvref_t<typename Problem::QDataType>;
//     using WDataType           = remove_cvref_t<typename Problem::WDataType>;
//     using S0accDataType       = remove_cvref_t<typename Problem::S0accDataType>;
//     using BiasDataType        = remove_cvref_t<typename Problem::BiasDataType>;
//     using GammaDataType       = remove_cvref_t<typename Problem::GammaDataType>;
//     using BetaDataType        = remove_cvref_t<typename Problem::BetaDataType>;
//     using RDataType           = remove_cvref_t<typename Problem::RDataType>;
//     using KDataType           = remove_cvref_t<typename Problem::KDataType>;
//     using VDataType           = remove_cvref_t<typename Problem::VDataType>;
//     using S1accDataType       = remove_cvref_t<typename Problem::S1accDataType>;
//     using KeyMaskDataType     = remove_cvref_t<typename Problem::KeyMaskDataType>;
//     using SMPLComputeDataType = remove_cvref_t<typename Problem::SMPLComputeDataType>;
//     using PDataType           = remove_cvref_t<typename Problem::PDataType>;
//     using OaccDataType        = remove_cvref_t<typename Problem::OaccDataType>;
//     using ODataType           = remove_cvref_t<typename Problem::ODataType>;

//     using BlockGemmLnAttnShape = remove_cvref_t<typename Problem::BlockGemmLnAttnShape>;

//     static constexpr index_t kBlockSize  = Problem::kBlockSize;
//     static constexpr index_t kBlockPerCu = 1;

//     static constexpr index_t kM0 = BlockGemmLnAttnShape::kM0;
//     static constexpr index_t kN0 = BlockGemmLnAttnShape::kN0;
//     static constexpr index_t kK0 = BlockGemmLnAttnShape::kK0;
//     static constexpr index_t kN1 = BlockGemmLnAttnShape::kN1;
//     static constexpr index_t kK1 = BlockGemmLnAttnShape::kK1;
//     static constexpr index_t kN2 = BlockGemmLnAttnShape::kN2;
//     static constexpr index_t kK2 = BlockGemmLnAttnShape::kK2;

//     static constexpr bool kPadBatchQ   = Problem::kPadBatchQ;
//     static constexpr bool kPadQWGemmK  = Problem::kPadQWGemmK;
//     static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
//     static constexpr bool kPadHeadDimQ = Problem::kPadHeadDimQ;
//     static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;

//     static constexpr index_t kAlignmentQ =
//         kPadQWGemmK ? 1 : Policy::template GetAlignmentQ<Problem>();
//     static constexpr index_t kAlignmentW =
//         kPadQWGemmK ? 1 : Policy::template GetAlignmentW<Problem>();
//     static constexpr index_t kAlignmentK =
//         kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
//     static constexpr index_t kAlignmentV =
//         kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();

//     static constexpr index_t kAlignmentBias =
//         kPadHeadDimQ ? 1 : Policy::template GetAlignmentBias<Problem>();
//     static constexpr index_t kAlignmentGamma =
//         kPadHeadDimQ ? 1 : Policy::template GetAlignmentGamma<Problem>();
//     static constexpr index_t kAlignmentBeta =
//         kPadHeadDimQ ? 1 : Policy::template GetAlignmentBeta<Problem>();
//     static constexpr index_t kAlignmentKeyMask =
//         kPadHeadDimQ ? 1 : Policy::template GetAlignmentKeyMask<Problem>();

//     static constexpr index_t kAlignmentO =
//         kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

//     static constexpr const char* name = "qwkvlds";

//     CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
//     {
//         return Policy::template GetSmemSize<Problem>();
//     }

//     template <typename QDramBlockWindowTmp,
//               typename WDramBlockWindowTmp,
//               typename BiasDramBlockWindowTmp,
//               typename GammaDramBlockWindowTmp,
//               typename BetaDramBlockWindowTmp,
//               typename KDramBlockWindowTmp,
//               typename KeyMaskDramBlockWindowTmp,
//               typename VDramBlockWindowTmp>
//     CK_TILE_HOST_DEVICE auto
//     operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
//                const WDramBlockWindowTmp& w_dram_block_window_tmp,
//                const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
//                const GammaDramBlockWindowTmp& gamma_dram_block_window_tmp,
//                const BetaDramBlockWindowTmp& beta_dram_block_window_tmp,
//                const KDramBlockWindowTmp& k_dram_block_window_tmp,
//                const KeyMaskDramBlockWindowTmp& keymask_dram_block_window_tmp,
//                const VDramBlockWindowTmp& v_dram_block_window_tmp,
//                const index_t K,
//                const index_t HDim_Q,
//                const index_t N1,
//                float scale_s,
//                float lrelu_alpha,
//                bool do_leaky_relu,
//                void* smem_ptr) const
//     {
//         static_assert(
//             std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
//                 std::is_same_v<WDataType, remove_cvref_t<typename WDramBlockWindowTmp::DataType>> &&
//                 std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
//                 std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
//             "wrong!");

//         static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
//                           kN0 == WDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
//                           kK0 == WDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
//                           kN1 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
//                           kK1 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
//                           kN2 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
//                           kK2 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
//                       "wrong!");

//         // Block GEMM
//         constexpr auto gemm_0 = Policy::template GetQWBlockGemm<Problem>();
//         constexpr auto gemm_1 = Policy::template GetRKBlockGemm<Problem>();
//         constexpr auto gemm_2 = Policy::template GetPVBlockGemm<Problem>();

//         // -------
//         // Stage 1
//         // R = LayerNorm(QW + Bias)
//         // [M0, N0] = LN([M0, K0] x [N0, K0] + [ , N0])
//         // Prepare
//         // -------

//         // Q: HBM -> LDS -> Reg
//         auto q_dram_window =
//             make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
//                              q_dram_block_window_tmp.get_window_lengths(),
//                              q_dram_block_window_tmp.get_window_origin(),
//                              Policy::template MakeQDramTileDistribution<Problem>());

//         QDataType* q_lds_ptr =
//             static_cast<QDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
//         auto q_lds = make_tensor_view<address_space_enum::lds>(
//             q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());
//         auto q_lds_window =
//             make_tile_window(q_lds, make_tuple(number<kM0>{}, number<kK0>{}), {0, 0});

//         // W: HBM -> LDS -> Reg
//         auto w_dram_window =
//             make_tile_window(w_dram_block_window_tmp.get_bottom_tensor_view(),
//                              w_dram_block_window_tmp.get_window_lengths(),
//                              w_dram_block_window_tmp.get_window_origin(),
//                              Policy::template MakeWDramTileDistribution<Problem>());

//         WDataType* w_lds_ptr = static_cast<WDataType*>(static_cast<void*>(
//             static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQ<Problem>()));
//         auto w_lds           = make_tensor_view<address_space_enum::lds>(
//             w_lds_ptr, Policy::template MakeWLdsBlockDescriptor<Problem>());
//         auto w_lds_window =
//             make_tile_window(w_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

//         // Bias/Gamma/Beta: HBM -> Reg
//         const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
//         auto bias_dram_window  = make_tile_window(
//             bias_dram_block_window_tmp.get_bottom_tensor_view(),
//             bias_dram_block_window_tmp.get_window_lengths(),
//             {bias_origin.at(number<0>{}), number<0>{}},
//             Policy::template MakeBiasGammaBetaDramTileDistribution<Problem, decltype(gemm_0)>());

//         const auto gamma_origin = gamma_dram_block_window_tmp.get_window_origin();
//         auto gamma_dram_window  = make_tile_window(
//             gamma_dram_block_window_tmp.get_bottom_tensor_view(),
//             gamma_dram_block_window_tmp.get_window_lengths(),
//             {gamma_origin.at(number<0>{}), number<0>{}},
//             Policy::template MakeBiasGammaBetaDramTileDistribution<Problem, decltype(gemm_0)>());

//         const auto beta_origin = beta_dram_block_window_tmp.get_window_origin();
//         auto beta_dram_window  = make_tile_window(
//             beta_dram_block_window_tmp.get_bottom_tensor_view(),
//             beta_dram_block_window_tmp.get_window_lengths(),
//             {beta_origin.at(number<0>{}), number<0>{}},
//             Policy::template MakeBiasGammaBetaDramTileDistribution<Problem, decltype(gemm_0)>());

//         using S0accBlockTileType = decltype(gemm_0.MakeCBlockTile());
//         auto s0_acc              = S0accBlockTileType{};

//         // -------
//         // Stage 1
//         // Start
//         // -------

//         // prefetch
//         auto q_block_tile = load_tile(q_dram_window);
//         move_tile_window(q_dram_window, {0, kK0});
//         auto w_block_tile = load_tile(w_dram_window);
//         move_tile_window(w_dram_window, {0, kK0});

//         store_tile(q_lds_window, q_block_tile);
//         store_tile(w_lds_window, w_block_tile);
//         clear_tile(s0_acc); // initialize C

//         index_t i_total_loops_gemm0     = 0;
//         const auto num_total_loop_gemm0 = integer_divide_ceil(K, kK0);

//         // hotloop
//         do
//         {
//             q_block_tile = load_tile(q_dram_window);
//             move_tile_window(q_dram_window, {0, kK0});
//             w_block_tile = load_tile(w_dram_window);
//             move_tile_window(w_dram_window, {0, kK0});

//             block_sync_lds();

//             gemm_0(s0_acc, q_lds_window, w_lds_window);

//             block_sync_lds();

//             store_tile(q_lds_window, q_block_tile);
//             store_tile(w_lds_window, w_block_tile);
//         } while(++i_total_loops_gemm0 < num_total_loop_gemm0 - 1);

//         // tail
//         auto bias  = load_tile(bias_dram_window);
//         auto gamma = load_tile(gamma_dram_window);
//         auto beta  = load_tile(beta_dram_window);

//         block_sync_lds();

//         gemm_0(s0_acc, q_lds_window, w_lds_window);

//         // bias
//         tile_elementwise_inout(
//             [&](auto& x, const auto& y) { x += type_convert<S0accDataType>(y); }, s0_acc, bias);

//         if(do_leaky_relu)
//         {
//             constexpr auto s0acc_spans = decltype(s0_acc)::get_distributed_spans();
//             sweep_tile_span(s0acc_spans[number<0>{}], [&](auto idx0) {
//                 sweep_tile_span(s0acc_spans[number<1>{}], [&](auto idx1) {
//                     constexpr auto i_j_idx = make_tuple(idx0, idx1);
//                     const auto val         = s0_acc[i_j_idx];

//                     s0_acc(i_j_idx) = (val > 0) ? val : val * lrelu_alpha;
//                 });
//             });
//         }

//         // layernorm
//         auto r_reg_tensor = make_static_distributed_tensor<RDataType>(
//             Policy::template MakeRRegBlockDescriptor<Problem, decltype(gemm_0)>());
//         constexpr S0accDataType epsilon = 1e-12;

//         auto intra_thread_count =
//             Policy::template GetLayerNormIntraLaneReduceCount<Problem, decltype(gemm_0)>(HDim_Q);

//         ThreadWelford<S0accDataType, RDataType> thread_welford{
//             type_convert<int>(intra_thread_count /* max_count per thread */)};

//         auto mean_compute_block_tensor =
//             thread_welford.template MakeInitialMeanVarDistributedTensor<decltype(r_reg_tensor)>();
//         auto var_compute_block_tensor =
//             thread_welford.template MakeInitialMeanVarDistributedTensor<decltype(r_reg_tensor)>();

//         clear_tile(mean_compute_block_tensor);
//         clear_tile(var_compute_block_tensor);

//         // Intra-thread reduce
//         thread_welford(s0_acc, mean_compute_block_tensor, var_compute_block_tensor);

//         // Intra-warp, Inter-thread reduce
//         WarpMergeWelford<S0accDataType, true>{}(
//             mean_compute_block_tensor, var_compute_block_tensor, thread_welford.cur_count_);

//         using InvSqrtBlockTileType = decltype(var_compute_block_tensor);

//         constexpr auto var_spans = InvSqrtBlockTileType::get_distributed_spans();

//         InvSqrtBlockTileType inv_std_compute_block_tensor;

//         sweep_tile_span(var_spans[number<0>{}], [&](auto idx0) {
//             constexpr auto i_idx = make_tuple(idx0);
//             inv_std_compute_block_tensor(i_idx) =
//                 type_convert<S0accDataType>(1.0f) /
//                 ck_tile::sqrt(var_compute_block_tensor[i_idx] + epsilon);
//         });

//         // Normalization
//         constexpr auto s0_acc_spans = decltype(s0_acc)::get_distributed_spans();

//         sweep_tile_span(s0_acc_spans[number<1>{}], [&](auto idx1) {
//             constexpr auto j_idx = make_tuple(idx1);
//             const auto gamma_    = type_convert<S0accDataType>(gamma[j_idx]);
//             const auto beta_     = type_convert<S0accDataType>(beta[j_idx]);

//             sweep_tile_span(s0_acc_spans[number<0>{}], [&](auto idx0) {
//                 constexpr auto i_idx   = make_tuple(idx0);
//                 constexpr auto i_j_idx = make_tuple(idx0, idx1);

//                 const auto mean    = mean_compute_block_tensor[i_idx];
//                 const auto inv_std = inv_std_compute_block_tensor[i_idx];

//                 const auto x = type_convert<S0accDataType>(s0_acc[i_j_idx]);
//                 auto y       = (x - mean) * inv_std * gamma_ + beta_;

//                 r_reg_tensor(i_j_idx) = type_convert<RDataType>(y);
//             });
//         });

//         // -------
//         // Stage 2
//         // P = Softmax(Masked[RK * scale])
//         // O = PV
//         // Prepare
//         // -------

//         // K: HBM ->LDS ->Reg
//         auto k_dram_window =
//             make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
//                              k_dram_block_window_tmp.get_window_lengths(),
//                              k_dram_block_window_tmp.get_window_origin(),
//                              Policy::template MakeKDramTileDistribution<Problem>());

//         KDataType* k_lds_ptr =
//             static_cast<KDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
//         auto k_lds = make_tensor_view<address_space_enum::lds>(
//             k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
//         auto k_lds_window =
//             make_tile_window(k_lds, make_tuple(number<kN1>{}, number<kK1>{}), {0, 0});

//         // V: HBM ->LDS ->Reg
//         auto v_dram_window =
//             make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
//                              v_dram_block_window_tmp.get_window_lengths(),
//                              v_dram_block_window_tmp.get_window_origin(),
//                              Policy::template MakeVDramTileDistribution<Problem>());

//         VDataType* v_lds_ptr =
//             static_cast<VDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
//         auto v_lds_write = make_tensor_view<address_space_enum::lds>(
//             v_lds_ptr, Policy::template MakeVLdsWriteBlockDescriptor<Problem>());
//         auto v_lds_write_window =
//             make_tile_window(v_lds_write, make_tuple(number<kK2>{}, number<kN2>{}), {0, 0});

//         auto v_lds_read = make_tensor_view<address_space_enum::lds>(
//             v_lds_ptr, Policy::template MakeVLdsReadBlockDescriptor<Problem>());
//         auto v_lds_read_window =
//             make_tile_window(v_lds_read, make_tuple(number<kN2>{}, number<kK2>{}), {0, 0});

//         // KeyMask
//         auto keymask_dram_window = make_tile_window(
//             keymask_dram_block_window_tmp.get_bottom_tensor_view(),
//             keymask_dram_block_window_tmp.get_window_lengths(),
//             {number<0>{}, number<0>{}},
//             Policy::template MakeKeyMaskDramTileDistribution<Problem, decltype(gemm_1)>());

//         using S1accBlockTileType = decltype(gemm_1.MakeCBlockTile());
//         auto s1_acc              = S1accBlockTileType{};

//         // reduction function for softmax
//         const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
//         const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

//         // infer Sacc, S, P, M, L, Oacc type
//         using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s1_acc));

//         using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
//             SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

//         using OaccBlockTileType = decltype(gemm_2.MakeCBlockTile());

//         // init Oacc, M, L
//         auto o_acc = OaccBlockTileType{};
//         auto m     = MLBlockTileType{};
//         auto l     = MLBlockTileType{};

//         clear_tile(o_acc);
//         set_tile(m, -numeric<SMPLComputeDataType>::infinity());
//         clear_tile(l);

//         const auto num_total_loop = integer_divide_ceil(N1, kN1);

//         // -------
//         // Stage 2
//         // Start
//         // -------

//         // prefetch K tile
//         index_t i_total_loops      = 0;
//         constexpr index_t k1_loops = kN0 / kK1;
//         constexpr index_t k2_loops = kN1 / kK2;

//         static_assert(1 <= k2_loops);
//         do
//         {
//             // STAGE 1, QK gemm
//             auto k_block_tile = load_tile(k_dram_window);
//             move_tile_window(k_dram_window, {0, kK1});
//             store_tile(k_lds_window, k_block_tile);
//             clear_tile(s1_acc);

//             // k_block_tile: correct
//             if constexpr(k1_loops > 1)
//             {
//                 static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
//                     k_block_tile = load_tile(k_dram_window);
//                     move_tile_window(k_dram_window, {0, kK1});

//                     block_sync_lds();
//                     gemm_1(s1_acc,
//                            get_slice_tile(r_reg_tensor,
//                                           sequence<0, i_k1 * kK1>{},
//                                           sequence<kM0, (i_k1 + 1) * kK1>{}),
//                            k_lds_window);
//                     block_sync_lds();

//                     store_tile(k_lds_window, k_block_tile);
//                 });
//             }

//             auto keymask = load_tile(keymask_dram_window);
//             move_tile_window(keymask_dram_window, {0, kN1});
//             auto v_block_tile = load_tile(v_dram_window); // prefetch load v tile

//             // tail
//             block_sync_lds();
//             gemm_1(s1_acc,
//                    get_slice_tile(r_reg_tensor,
//                                   sequence<0, (k1_loops - 1) * kK1>{},
//                                   sequence<kM0, k1_loops * kK1>{}),
//                    k_lds_window);
//             block_sync_lds();

//             // STAGE 2, scale_s, mask, softmax
//             constexpr auto s1_spans = decltype(s1_acc)::get_distributed_spans();
//             sweep_tile_span(s1_spans[number<1>{}], [&](auto idx1) {
//                 constexpr auto j_idx = make_tuple(idx1);
//                 auto keymask_val     = keymask[j_idx];

//                 sweep_tile_span(s1_spans[number<0>{}], [&](auto idx0) {
//                     constexpr auto i_j_idx = make_tuple(idx0, idx1);

//                     s1_acc(i_j_idx) =
//                         keymask_val ? s1_acc[i_j_idx] : -numeric<SMPLComputeDataType>::infinity();
//                 });
//             });

//             const auto s = cast_tile<SMPLComputeDataType>(s1_acc);
//             auto m_local = block_tile_reduce<SMPLComputeDataType>(
//                 s, sequence<1>{}, f_max, -numeric<SMPLComputeDataType>::infinity());
//             block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

//             const auto m_old = m;
//             tile_elementwise_inout(
//                 [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);

//             auto p_compute =
//                 make_static_distributed_tensor<SMPLComputeDataType>(s.get_tile_distribution());

//             constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
//             sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
//                 constexpr auto i_idx = make_tuple(idx0);

//                 auto row_max = scale_s * m[i_idx];

//                 sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
//                     constexpr auto i_j_idx = make_tuple(idx0, idx1);

//                     p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
//                 });
//             });

//             auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
//                 p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0});

//             block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
//             // l{j}, Oacc{j}
//             constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
//             sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
//                 constexpr auto i_idx = make_tuple(idx0);

//                 const auto tmp = [&]() {
//                     auto row_max = scale_s * m[i_idx];
//                     return exp2(scale_s * m_old[i_idx] - row_max);
//                 }();

//                 l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
//                 sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
//                     constexpr auto i_j_idx = make_tuple(idx0, idx1);

//                     o_acc(i_j_idx) *= tmp;
//                 });
//             });

//             block_sync_lds();

//             auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
//                 Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
//             shuffle_tile(v_shuffle_tmp, v_block_tile);
//             store_tile(v_lds_write_window, v_shuffle_tmp); // store the prefetch

//             move_tile_window(v_dram_window, {kK2, 0});

//             const auto p = cast_tile<PDataType>(p_compute);

//             // STAGE 3, PV gemm
//             if constexpr(k2_loops > 1)
//             {
//                 static_for<0, k2_loops - 1, 1>{}([&](auto i_k2) {
//                     const auto v = load_tile(v_dram_window); // load next v
//                     block_sync_lds();
//                     gemm_2(o_acc,
//                            get_slice_tile(
//                                p, sequence<0, i_k2 * kK2>{}, sequence<kM0, (i_k2 + 1) * kK2>{}),
//                            v_lds_read_window);
//                     block_sync_lds();

//                     v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
//                         Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
//                     shuffle_tile(v_shuffle_tmp, v);
//                     store_tile(v_lds_write_window, v_shuffle_tmp); // store the prefetch

//                     move_tile_window(v_dram_window, {kK2, 0});
//                 });
//             }
//             // move K tile windows
//             move_tile_window(k_dram_window, {kN1, -kN0});
//             // tail
//             {
//                 block_sync_lds();
//                 gemm_2(o_acc,
//                        get_slice_tile(p, sequence<0, (k2_loops - 1) * kK2>{}, sequence<kM0, kN1>{}),
//                        v_lds_read_window);
//                 block_sync_lds();
//             }
//         } while(++i_total_loops < num_total_loop);

//         // finally, O
//         constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

//         sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
//             constexpr auto i_idx = make_tuple(idx0);
//             const auto tmp       = (l[i_idx] == 0.f) ? 0.f : 1 / l[i_idx];

//             sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
//                 constexpr auto i_j_idx = make_tuple(idx0, idx1);
//                 o_acc(i_j_idx) *= tmp;
//             });
//         });

//         return o_acc;
//     }
// };

// } // namespace ck_tile
