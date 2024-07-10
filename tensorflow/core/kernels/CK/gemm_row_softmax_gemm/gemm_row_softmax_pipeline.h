// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/reduce/block/block_reduce.hpp>

#include "gemm_row_softmax_gemm_pipeline_policy.h"
#include "gemm_row_softmax_gemm_problem.h"

namespace ck_tile {

template <typename GemmRowSoftmaxGemmProblem_, typename GemmRowSoftmaxTraits_>
struct BlockGemmRowSoftmaxPipeline {
  using Problem = remove_cvref_t<GemmRowSoftmaxGemmProblem_>;
  using Traits = remove_cvref_t<GemmRowSoftmaxTraits_>;

  using Policy = GemmRowSoftmaxGemmPolicy;

  using InOutDataType = remove_cvref_t<typename Problem::InOutDataType>;
  using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;
  // DataType used when computing Softmax
  using SMComputeDataType = remove_cvref_t<typename Problem::SMComputeDataType>;
  using MaskDataType = remove_cvref_t<typename Problem::MaskDataType>;

  using GemmRowSoftmaxTileShape =
      remove_cvref_t<typename Problem::GemmRowSoftmaxTileShape>;

  static constexpr index_t kM = GemmRowSoftmaxTileShape::kGemmM;
  static constexpr index_t kN = GemmRowSoftmaxTileShape::kGemmN;
  static constexpr index_t kK = GemmRowSoftmaxTileShape::kGemmK;
  static constexpr index_t kMaxK = GemmRowSoftmaxTileShape::kMaxK;

  static constexpr bool kPadSeqLen = Traits::kPadGemmN;
  static constexpr bool kPadHeadDim = Traits::kPadGemmK;
  static constexpr bool kPadNewHeadDim = Traits::kPadGemmM;

  static constexpr index_t kAlignmentB0 = 1;
  static constexpr index_t kAlignmentA0 = 1;
  static constexpr index_t kAlignmentB1 = 1;
  static constexpr index_t kAlignmentMask = 1;

  static constexpr index_t kBlockSize = Problem::kGemm0BlockSize;
  static constexpr index_t kBlockPerCu = Traits::kGemmBlockPerCu;

  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
    return Policy::GetGemmRowSoftmaxKernelSmemSize<Problem>();
  }

  template <typename B0TDramBlockWindowTmp, typename A0DramBlockWindowTmp,
            typename MaskDramBlockWindowTmp, typename B1DramBlockWindowTmp>
  CK_TILE_HOST_DEVICE auto operator()(
      const B0TDramBlockWindowTmp& b0t_dram_block_window_tmp,
      const A0DramBlockWindowTmp& a0_dram_block_window_tmp,
      const MaskDramBlockWindowTmp& mask_dram_block_window_tmp,
      B1DramBlockWindowTmp& b1_dram_block_window_tmp, index_t seqlen,
      void* smem_ptr) const {
    static_assert(
        std::is_same_v<
            InOutDataType,
            remove_cvref_t<typename B0TDramBlockWindowTmp::DataType>> &&
            std::is_same_v<
                InOutDataType,
                remove_cvref_t<typename A0DramBlockWindowTmp::DataType>> &&
            std::is_same_v<
                InOutDataType,
                remove_cvref_t<typename B1DramBlockWindowTmp::DataType>> &&
            std::is_same_v<
                MaskDataType,
                remove_cvref_t<typename MaskDramBlockWindowTmp::DataType>>,
        "wrong!");

    static_assert(
        kN == B0TDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kM == A0DramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kMaxK ==
                B0TDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
            kMaxK == A0DramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
            kM == B1DramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kN == B1DramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
        "wrong!");

    // A0 tile in LDS
    InOutDataType* a0_lds_ptr =
        static_cast<InOutDataType*>(static_cast<void*>(smem_ptr));
    auto a0_lds = make_tensor_view<address_space_enum::lds>(
        a0_lds_ptr, Policy::template MakeA0LdsBlockDescriptor<Problem>());
    auto a0_lds_window = make_tile_window(
        a0_lds, make_tuple(number<kM>{}, number<kMaxK>{}), {0, 0});

    // Block GEMM
    constexpr auto gemm = Policy::template GetA0B0TBlockGemm<Problem>();

    auto a0_dram_window = make_tile_window(
        a0_dram_block_window_tmp.get_bottom_tensor_view(),
        a0_dram_block_window_tmp.get_window_lengths(),
        a0_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeA0DramTileDistribution<Problem, decltype(gemm)>());

    auto b0t_dram_window = make_tile_window(
        b0t_dram_block_window_tmp.get_bottom_tensor_view(),
        b0t_dram_block_window_tmp.get_window_lengths(),
        b0t_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeB0TDramTileDistribution<Problem,
                                                     decltype(gemm)>());

    auto mask_dram_window = make_tile_window(
        mask_dram_block_window_tmp.get_bottom_tensor_view(),
        mask_dram_block_window_tmp.get_window_lengths(),
        mask_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeMaskDramTileDistribution<Problem,
                                                      decltype(gemm)>());

    auto b1_dram_window = b1_dram_block_window_tmp;

    // Load in once a0 from global to lds
    auto a0_tile = load_tile(a0_dram_window);
    store_tile(a0_lds_window, a0_tile);

    block_sync_lds();

    using AccBlockTileType = decltype(gemm.MakeCBlockTile());
    auto s_acc = AccBlockTileType{};

    // reduction function for softmax
    const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

    using SBlockTileType = decltype(cast_tile<SMComputeDataType>(s_acc));

    using MLBlockTileType = decltype(block_tile_reduce<SMComputeDataType>(
        SBlockTileType{}, sequence<1>{}, f_max, SMComputeDataType{0}));

    // init M, L
    auto m = MLBlockTileType{};
    auto l = MLBlockTileType{};

    clear_tile(l);
    set_tile(m, -numeric<SMComputeDataType>::infinity());

    constexpr auto k_loops = kMaxK / kK;

    const index_t num_total_loop = integer_divide_ceil(seqlen, kN);

    index_t i_main_loop = 0;

    // first main-loop calculate the global max and sum-of-exp
    do {
      auto b0t_tile = load_tile(b0t_dram_window);
      auto mask = load_tile(mask_dram_window);

      s_acc = gemm(
          get_slice_tile(a0_lds_window, sequence<0, 0>{}, sequence<kM, kK>{}),
          get_slice_tile(b0t_tile, sequence<0, 0>{}, sequence<kN, kK>{}));

      static_for<1, k_loops, 1>{}([&](auto i_k) {
        gemm(s_acc,
             get_slice_tile(a0_lds_window, sequence<0, i_k * kK>{},
                            sequence<kM, (i_k + 1) * kK>{}),
             get_slice_tile(b0t_tile, sequence<0, i_k * kK>{},
                            sequence<kN, (i_k + 1) * kK>{}));
      });

      move_tile_window(b0t_dram_window, {kN, 0});
      move_tile_window(mask_dram_window, {kN});

      tile_elementwise_inout([&](auto& x) { x = (x > 0) ? x : 0; }, s_acc);

      auto s = cast_tile<SMComputeDataType>(s_acc);  // S{j}

      constexpr auto s_spans = decltype(s)::get_distributed_spans();

      sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);
          constexpr auto j_idx = make_tuple(idx1);

          s(i_j_idx) = mask[j_idx] ? s[i_j_idx]
                                   : -numeric<SMComputeDataType>::infinity();
        });
      });

      // reduce to get the local maximum
      // ToDo: a cross-warp reduce is needed to support NWarps > 1
      auto m_local = block_tile_reduce<SMComputeDataType>(
          s, sequence<1>{}, f_max,
          -numeric<SMComputeDataType>::infinity());  // m_local = rowmax(S{j})
      block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

      // update the global maximum value
      const auto m_old = m;  //
      tile_elementwise_inout(
          [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old,
          m_local);  // m{j}

      // get the stabilized exp() of the current tile using the latest global
      // maximum value
      sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);

        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);
          s(i_j_idx) = exp(s[i_j_idx] - m[i_idx]);
        });
      });

      // calculate the sum-of-exp for the current tile
      auto rowsum_s = block_tile_reduce<SMComputeDataType>(
          s, sequence<1>{}, f_sum,
          SMComputeDataType{0});  // rowsum(Pcompute{j})

      block_tile_reduce_sync(rowsum_s, f_sum, bool_constant<false>{});

      // update the global sum-of-exp
      sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);
        const auto tmp = exp(m_old[i_idx] - m[i_idx]);
        l(i_idx) = tmp * l[i_idx] + rowsum_s[i_idx];
      });
    } while (++i_main_loop < num_total_loop);

    // reset the window location and loop index
    move_tile_window(b0t_dram_window, {-kN * num_total_loop, 0});
    move_tile_window(mask_dram_window, {-kN * num_total_loop});

    i_main_loop = 0;

    // second main-loop to normalize the gemm result using global max and
    // sum-of-exp
    do {
      auto b0t_tile = load_tile(b0t_dram_window);
      auto mask = load_tile(mask_dram_window);

      s_acc = gemm(
          get_slice_tile(a0_lds_window, sequence<0, 0>{}, sequence<kM, kK>{}),
          get_slice_tile(b0t_tile, sequence<0, 0>{}, sequence<kN, kK>{}));

      static_for<1, k_loops, 1>{}([&](auto i_k) {
        gemm(s_acc,
             get_slice_tile(a0_lds_window, sequence<0, i_k * kK>{},
                            sequence<kM, (i_k + 1) * kK>{}),
             get_slice_tile(b0t_tile, sequence<0, i_k * kK>{},
                            sequence<kN, (i_k + 1) * kK>{}));
      });

      tile_elementwise_inout([&](auto& x) { x = (x > 0) ? x : 0; }, s_acc);

      auto s = cast_tile<SMComputeDataType>(s_acc);  // S{j}

      constexpr auto s_spans = decltype(s)::get_distributed_spans();

      sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
          constexpr auto j_idx = make_tuple(idx1);
          constexpr auto i_j_idx = make_tuple(idx0, idx1);

          s(i_j_idx) = mask[j_idx] ? s[i_j_idx]
                                   : -numeric<SMComputeDataType>::infinity();
        });
      });

      sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);

        sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);
          if (l[i_idx] == 0.f)
            s(i_j_idx) =
                1.0f / ck_tile::type_convert<SMComputeDataType>(seqlen);
          else
            s(i_j_idx) = exp(s[i_j_idx] - m[i_idx]) / l[i_idx];
        });
      });

      const auto b1_tile = cast_tile<InOutDataType>(s);

      store_tile(b1_dram_window, b1_tile);

      move_tile_window(b0t_dram_window, {kN, 0});
      move_tile_window(mask_dram_window, {kN});
      move_tile_window(b1_dram_window, {0, kN});
    } while (++i_main_loop < num_total_loop);
  };
};

}  // namespace ck_tile
