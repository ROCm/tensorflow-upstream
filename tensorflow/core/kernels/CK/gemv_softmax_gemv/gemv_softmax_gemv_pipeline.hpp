// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/reduce/block/block_reduce.hpp>

#include "gemv_softmax_gemv_pipeline_policy.hpp"
#include "gemv_softmax_gemv_problem.hpp"

namespace ck_tile {

template <typename GemvSoftmaxGemvProblem_, typename GemvSoftmaxGemvTraits_>
struct BlockGemvSoftmaxGemvPipeline {
  using Problem = remove_cvref_t<GemvSoftmaxGemvProblem_>;
  using Traits = remove_cvref_t<GemvSoftmaxGemvTraits_>;

  using Policy = GemvSoftmaxGemvPolicy;

  using InOutDataType = remove_cvref_t<typename Problem::InOutDataType>;
  using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;
  // DataType used when computing Softmax
  using SMPLComputeDataType =
      remove_cvref_t<typename Problem::SMPLComputeDataType>;
  using MaskDataType = remove_cvref_t<typename Problem::MaskDataType>;

  // export to be used by the kernel
  using GemvSoftmaxGemvTileShape =
      remove_cvref_t<typename Problem::GemvSoftmaxGemvTileShape>;

  static constexpr index_t kM = GemvSoftmaxGemvTileShape::kM;
  static constexpr index_t kN0 = GemvSoftmaxGemvTileShape::kN0;
  static constexpr index_t kK0 = GemvSoftmaxGemvTileShape::kK0;
  static constexpr index_t kN1 = GemvSoftmaxGemvTileShape::kN1;
  static constexpr index_t kK1 = GemvSoftmaxGemvTileShape::kK1;
  static constexpr index_t kMaxK = GemvSoftmaxGemvTileShape::kMaxK;

  static_assert(
      kN1 == kMaxK,
      "This pipeline requies head_sz of b1 not split during the processing!");

  // export to be used by the kernel
  static constexpr bool kPadSeqLen = Traits::kPadSeqLen;
  static constexpr bool kPadAB0HeadSz = Traits::kPadAB0HeadSz;
  static constexpr bool kPadB1DHeadSz = Traits::kPadB1DHeadSz;

  // export to be used by the kernel
  static constexpr index_t kAlignmentA = 1;
  static constexpr index_t kAlignmentB0 = 1;
  static constexpr index_t kAlignmentB1 = 1;
  static constexpr index_t kAlignmentMask = 1;
  static constexpr index_t kAlignmentD = 1;

  // export to be used by the kernel
  static constexpr index_t kBlockSize = Problem::kBlockSize;
  static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;

  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
    return Policy::GetGemvSoftmaxGemvSmemSize<Problem>();
  }

  template <typename ADramBlockWindowTmp, typename B0DramBlockWindowTmp,
            typename MaskDramBlockWindowTmp, typename B1TDramBlockWindowTmp>
  CK_TILE_HOST_DEVICE auto operator()(
      const ADramBlockWindowTmp& a_dram_block_window_tmp,
      const B0DramBlockWindowTmp& b0_dram_block_window_tmp,
      const MaskDramBlockWindowTmp& mask_dram_block_window_tmp,
      const B1TDramBlockWindowTmp& b1t_dram_block_window_tmp, index_t seqlen,
      index_t head_sz, void* smem_ptr) const {
    static_assert(
        std::is_same_v<
            InOutDataType,
            remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
            std::is_same_v<
                InOutDataType,
                remove_cvref_t<typename B0DramBlockWindowTmp::DataType>> &&
            std::is_same_v<
                InOutDataType,
                remove_cvref_t<typename B1TDramBlockWindowTmp::DataType>> &&
            std::is_same_v<
                MaskDataType,
                remove_cvref_t<typename MaskDramBlockWindowTmp::DataType>> &&
            std::is_same_v<
                MaskDataType,
                remove_cvref_t<typename MaskDramBlockWindowTmp::DataType>>,
        "wrong!");

    static_assert(
        kMaxK == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
            kN0 == B0DramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kK0 == B0DramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
            kN1 == B1TDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kK1 == B1TDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
        "wrong!");

    // b0 tile in LDS
    auto b0_lds = make_tensor_view<address_space_enum::lds>(
        static_cast<InOutDataType*>(smem_ptr),
        Policy::template MakeB0LdsBlockDescriptor<Problem>());
    auto b0_lds_window = make_tile_window(
        b0_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

    // b1t tile in LDS
    InOutDataType* b1t_lds_ptr = reinterpret_cast<InOutDataType*>(
        static_cast<char*>(smem_ptr) +
        Policy::template GetSmemSizeB0<Problem>());
    auto b1t_lds = make_tensor_view<address_space_enum::lds>(
        b1t_lds_ptr, Policy::template MakeB1TLdsBlockDescriptor<Problem>());
    auto b1t_lds_window = make_tile_window(
        b1t_lds, make_tuple(number<kN1>{}, number<kK1>{}), {0, 0});

    // Block GEMM
    constexpr auto gemm_0 = Policy::template GetAB0BlockGemm<Problem>();
    constexpr auto gemm_1 = Policy::template GetPB1TBlockGemm<Problem>();

    auto mask_dram_window = make_tile_window(
        mask_dram_block_window_tmp.get_bottom_tensor_view(),
        mask_dram_block_window_tmp.get_window_lengths(),
        mask_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeMaskDramTileDistribution<Problem,
                                                      decltype(gemm_0)>());

    auto a_dram_window = make_tile_window(
        a_dram_block_window_tmp.get_bottom_tensor_view(),
        a_dram_block_window_tmp.get_window_lengths(),
        a_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeADramTileDistribution<Problem,
                                                   decltype(gemm_0)>());

    // load in once whole tile of length kMaxK in head_sz dimension
    auto a_tile = load_tile(a_dram_window);

    using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
    auto s_acc = SaccBlockTileType{};

    // reduction function for softmax
    const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

    // infer Sacc, S, P, M, L, Dacc type
    using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

    using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
        SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

    using DaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

    // init Dacc, M, L
    auto d_acc = DaccBlockTileType{};
    auto m = MLBlockTileType{};
    auto l = MLBlockTileType{};

    clear_tile(d_acc);
    set_tile(m, -numeric<SMPLComputeDataType>::infinity());
    clear_tile(l);

    const auto num_total_loop = integer_divide_ceil(seqlen, kN0);

    // b0 window which moves along the seqlen dimension in the main-loop
    auto b0_dram_main_window = b0_dram_block_window_tmp;

    // b1t window which moves along seqlen dimension in both the main-loop and
    // k1-loop
    auto b1t_dram_window = make_tile_window(
        b1t_dram_block_window_tmp.get_bottom_tensor_view(),
        b1t_dram_block_window_tmp.get_window_lengths(),
        b1t_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeB1TDramTileDistribution<Problem,
                                                     decltype(gemm_1)>());

    index_t i_total_loops = 0;
    constexpr index_t k0_loops = kMaxK / kK0;
    constexpr index_t k1_loops = kN0 / kK1;

    static_assert(2 <= k0_loops);
    static_assert(1 <= k1_loops);

    const GemmAccDataType mul =
        ck_tile::type_convert<GemmAccDataType>(1.0f) /
        ck_tile::sqrt(ck_tile::type_convert<GemmAccDataType>(head_sz));

    do {
      auto mask = load_tile(mask_dram_window);

      // ---- Stage 1: A @ B0 Gemm ----

      // b0 window which moves along the head_sz dimension during gemm_0
      auto b0_dram_k0_window = make_tile_window(
          b0_dram_main_window.get_bottom_tensor_view(),
          b0_dram_main_window.get_window_lengths(),
          b0_dram_main_window.get_window_origin(),
          Policy::template MakeB0DramTileDistribution<Problem,
                                                      decltype(gemm_0)>());

      // load b0 tile for first iteration
      auto b0_block_tile = load_tile(b0_dram_k0_window);
      {
        move_tile_window(b0_dram_k0_window, {0, kK0});

        clear_tile(s_acc);  // initialize C

        // store b0 tile to LDS for first iteration
        store_tile(b0_lds_window, b0_block_tile);

        // load b0 tile for second iteration
        b0_block_tile = load_tile(b0_dram_k0_window);
      }

      if constexpr (k0_loops > 2) {
        static_for<0, k0_loops - 2, 1>{}([&](auto i_k0) {
          block_sync_lds();
          gemm_0(s_acc,
                 get_slice_tile(a_tile, sequence<0, i_k0 * kK0>{},
                                sequence<kM, (i_k0 + 1) * kK0>{}),
                 b0_lds_window);

          move_tile_window(b0_dram_k0_window, {0, kK0});

          // store b0 tile to LDS for next iteration
          block_sync_lds();
          store_tile(b0_lds_window, b0_block_tile);

          // store b0 tile to LDS for next+1 iteration
          b0_block_tile = load_tile(b0_dram_k0_window);
        });
      }

      // in-advance load b1t tile for first iteration
      const auto b1t_prefetch = load_tile(b1t_dram_window);

      // last two iterations of gemm_0
      {
        block_sync_lds();
        gemm_0(s_acc,
               get_slice_tile(a_tile, sequence<0, (k0_loops - 2) * kK0>{},
                              sequence<kM, (k0_loops - 1) * kK0>{}),
               b0_lds_window);

        // store b0 tile to LDS for last iteration
        block_sync_lds();
        store_tile(b0_lds_window, b0_block_tile);

        block_sync_lds();
        gemm_0(s_acc,
               get_slice_tile(a_tile, sequence<0, (k0_loops - 1) * kK0>{},
                              sequence<kM, k0_loops * kK0>{}),
               b0_lds_window);
      }

      auto s = cast_tile<SMPLComputeDataType>(s_acc);

      constexpr auto s_spans = decltype(s)::get_distributed_spans();

      // masking
      sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
          constexpr auto j_idx = make_tuple(idx1);
          constexpr auto i_j_idx = make_tuple(idx0, idx1);

          s(i_j_idx) = mask[j_idx] ? s[i_j_idx] * mul
                                   : -numeric<SMPLComputeDataType>::infinity();
        });
      });

      // ---- Stage 2: Softmax ----

      // reduce to get local max of current tile
      auto m_local = block_tile_reduce<SMPLComputeDataType>(
          s, sequence<1>{}, f_max, -numeric<SMPLComputeDataType>::infinity());
      block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

      const auto m_old = m;
      tile_elementwise_inout(
          [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old,
          m_local);

      auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
          s.get_tile_distribution());

      // calculate local stabilized exp()
      constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
      sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);
        sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);
          if (m[i_idx] == -numeric<SMPLComputeDataType>::infinity())
            p_compute(i_j_idx) =
                ck_tile::type_convert<SMPLComputeDataType>(0.0f);
          else
            p_compute(i_j_idx) = exp(s[i_j_idx] - m[i_idx]);
        });
      });

      // reduce to get local sum of stabilized exp()
      auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
          p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0});
      block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

      // calculate the global sum of stabilized exp(), and
      // adjust to the previous gemm_1 result accordingly
      constexpr auto d_spans = decltype(d_acc)::get_distributed_spans();
      sweep_tile_span(d_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);

        if (m[i_idx] != -numeric<SMPLComputeDataType>::infinity()) {
          const auto tmp = exp(m_old[i_idx] - m[i_idx]);
          l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];

          sweep_tile_span(d_spans[number<1>{}], [&](auto idx1) {
            constexpr auto i_j_idx = make_tuple(idx0, idx1);
            d_acc(i_j_idx) *= tmp;
          });
        };
      });

      // store b1t tile to LDS for first iteration
      block_sync_lds();
      store_tile(b1t_lds_window, b1t_prefetch);

      move_tile_window(b1t_dram_window, {0, kK1});

      const auto p = cast_tile<InOutDataType>(p_compute);

      // ---- Stage 3: P @ B1T gemm ----
      if constexpr (k1_loops > 1) {
        static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
          // load b1t tile for next iteration
          const auto b1t = load_tile(b1t_dram_window);

          block_sync_lds();
          gemm_1(d_acc,
                 get_slice_tile(p, sequence<0, i_k1 * kK1>{},
                                sequence<kM, (i_k1 + 1) * kK1>{}),
                 b1t_lds_window);

          // store b1t tile to LDS for next iteration
          block_sync_lds();
          store_tile(b1t_lds_window, b1t);

          move_tile_window(b1t_dram_window, {0, kK1});
        });
      }

      move_tile_window(b0_dram_main_window, {kN0, 0});
      move_tile_window(mask_dram_window, {kN0});

      // tail iteration of gemm_1
      {
        block_sync_lds();
        gemm_1(d_acc,
               get_slice_tile(p, sequence<0, (k1_loops - 1) * kK1>{},
                              sequence<kM, kN0>{}),
               b1t_lds_window);

        block_sync_lds();
      }
    } while (++i_total_loops < num_total_loop);

    // final d
    constexpr auto d_spans = decltype(d_acc)::get_distributed_spans();

    sweep_tile_span(d_spans[number<0>{}], [&](auto idx0) {
      constexpr auto i_idx = make_tuple(idx0);
      if (l[i_idx] != ck_tile::type_convert<SMPLComputeDataType>(0.0f)) {
        sweep_tile_span(d_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);
          d_acc(i_j_idx) *= 1 / l[i_idx];
        });
      };
    });

    // The folllowing codes is for handling the special case where all seqlen
    // values are -inf

    auto p_gemm = make_static_distributed_tensor<InOutDataType>(
        s_acc.get_tile_distribution());

    bool row_all_inf_occurred = false;
    {
      constexpr auto p_spans = decltype(p_gemm)::get_distributed_spans();
      sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);

        if (l[i_idx] == ck_tile::type_convert<SMPLComputeDataType>(0.0f)) {
          row_all_inf_occurred = true;
          sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
            constexpr auto i_j_idx = make_tuple(idx0, idx1);
            p_gemm(i_j_idx) = ck_tile::type_convert<InOutDataType>(
                1.0f / static_cast<float>(seqlen));
          });
        } else {
          sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
            constexpr auto i_j_idx = make_tuple(idx0, idx1);
            p_gemm(i_j_idx) = ck_tile::type_convert<InOutDataType>(0.0f);
          });
        };
      });
    }

    if (row_all_inf_occurred) {
      i_total_loops = 0;

      move_tile_window(b1t_dram_window, {0, -kN0 * num_total_loop});

      do {
        // in-advance load b1t tile for first iteration
        const auto b1t_prefetch = load_tile(b1t_dram_window);
        // store b1t tile to LDS for first iteration
        block_sync_lds();
        store_tile(b1t_lds_window, b1t_prefetch);

        move_tile_window(b1t_dram_window, {0, kK1});

        // ---- Stage 3: P @ B1T gemm ----
        if constexpr (k1_loops > 1) {
          static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
            // load b1t tile for next iteration
            const auto b1t = load_tile(b1t_dram_window);

            block_sync_lds();
            gemm_1(d_acc,
                   get_slice_tile(p_gemm, sequence<0, i_k1 * kK1>{},
                                  sequence<kM, (i_k1 + 1) * kK1>{}),
                   b1t_lds_window);

            // store b1t tile to LDS for next iteration
            block_sync_lds();
            store_tile(b1t_lds_window, b1t);
            move_tile_window(b1t_dram_window, {0, kK1});
          });
        }

        // tail iteration of gemm_1
        {
          block_sync_lds();
          gemm_1(d_acc,
                 get_slice_tile(p_gemm, sequence<0, (k1_loops - 1) * kK1>{},
                                sequence<kM, kN0>{}),
                 b1t_lds_window);
        }
      } while (++i_total_loops < num_total_loop);
    }

    return d_acc;
  };
};

}  // namespace ck_tile
