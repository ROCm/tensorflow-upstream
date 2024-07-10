// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "bool_switch.hpp"
#include "gather_attention_kernel.hpp"
#include "gather_attention_params.hpp"
#include "gather_attention_pipeline.hpp"
#include "gather_attention_problem.hpp"
#include "gather_attention_setting.hpp"
#include "gather_attention_tile_traits.hpp"

template <typename InOutDataType, typename MaskDataType, ck_tile::index_t kMaxK>
struct gather_attention_dispatch {
  using GatherAttentionProblem_ = ck_tile::GatherAttentionProblem<
      InOutDataType,
      typename GatherAttentionTypeConfig<InOutDataType>::GemmAccDataType,
      typename GatherAttentionTypeConfig<InOutDataType>::SMComputeDataType,
      MaskDataType, GatherAttentionTileShape<kMaxK>>;

  static void Run(const GatherAttentionParams& param, hipStream_t stream) {
    using GatherAttentionTileShape_ = GatherAttentionTileShape<kMaxK>;

    const bool pad_seqlen =
        !(param.seqlen % GatherAttentionTileShape_::kN0 == 0);
    const bool pad_ab0_head_sz =
        !(param.head_sz % GatherAttentionTileShape_::kMaxK == 0);
    const bool pad_b1d_head_sz =
        !(param.head_sz % GatherAttentionTileShape_::kN1 == 0);

    BOOL_SWITCH_3(
        pad_seqlen, kPadSeqLen, pad_ab0_head_sz, kPadAB0HeadSz, pad_b1d_head_sz,
        kPadB1DHeadSz, [&] {
          using GatherAttentionTraits_ =
              ck_tile::GatherAttentionTileTraits<kPadSeqLen, kPadAB0HeadSz,
                                                 kPadB1DHeadSz, 1>;

          using GatherAttentionPipeline =
              ck_tile::BlockGatherAttentionPipeline<GatherAttentionProblem_,
                                                    GatherAttentionTraits_>;

          using GatherAttentionEpilogue =
              ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                  typename GatherAttentionTypeConfig<
                      InOutDataType>::GemmAccDataType,
                  InOutDataType, true, kPadB1DHeadSz>>;

          using GatherAttentionKernel =
              ck_tile::GatherAttentionKernel<GatherAttentionPipeline,
                                             GatherAttentionEpilogue>;

          RunKernel<GatherAttentionKernel>(param, stream);
        });
  };

  template <typename GatherAttentionKernel>
  static void RunKernel(const GatherAttentionParams& param,
                        hipStream_t stream) {
    const auto kargs = [&] {
      return GatherAttentionKernel::MakeKargs(
          param.a_ptr, param.b0_ptr, param.mask_ptr, param.indices_ptr,
          param.b1_ptr, param.d_ptr, param.num_index, param.seqlen,
          param.num_head, param.head_sz, param.a_batch_stride,
          param.a_nhead_stride, param.b0_batch_stride, param.b0_seq_stride,
          param.b0_nhead_stride, param.mask_batch_stride, param.b1_batch_stride,
          param.b1_seq_stride, param.b1_nhead_stride, param.d_batch_stride,
          param.d_nhead_stride);
    }();

    dim3 kGridSize = GatherAttentionKernel::GridSize(&kargs);
    constexpr dim3 kBlockSize = GatherAttentionKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = GatherAttentionKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            GatherAttentionKernel{}, kGridSize, kBlockSize, 0, kargs));
  };
};

template <typename InOutDataType, typename MaskDataType, ck_tile::index_t kMaxK>
void run_gather_attention(const GatherAttentionParams& param,
                          hipStream_t stream) {
  gather_attention_dispatch<InOutDataType, MaskDataType, kMaxK>::Run(param,
                                                                     stream);
};
