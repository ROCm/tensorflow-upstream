// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "bool_switch.hpp"
#include "gemv_softmax_gemv_kernel.hpp"
#include "gemv_softmax_gemv_params.hpp"
#include "gemv_softmax_gemv_pipeline.hpp"
#include "gemv_softmax_gemv_problem.hpp"
#include "gemv_softmax_gemv_setting.hpp"
#include "gemv_softmax_gemv_tile_traits.hpp"

template <typename InOutDataType, typename MaskDataType, ck_tile::index_t kMaxK>
struct gemv_softmax_gemv_dispatch {
  using GemvSoftmaxGemvProblem_ = ck_tile::GemvSoftmaxGemvProblem<
      InOutDataType,
      typename GemvSoftmaxGemvTypeConfig<InOutDataType>::GemmAccDataType,
      typename GemvSoftmaxGemvTypeConfig<InOutDataType>::SMComputeDataType,
      MaskDataType, GemvSoftmaxGemvTileShape<kMaxK>>;

  static void Run(const GemvSoftmaxGemvParams& param, hipStream_t stream) {
    using GemvSoftmaxGemvTileShape_ = GemvSoftmaxGemvTileShape<kMaxK>;

    const bool pad_seqlen =
        !(param.seqlen % GemvSoftmaxGemvTileShape_::kN0 == 0);
    const bool pad_ab0_head_sz =
        !(param.head_sz % GemvSoftmaxGemvTileShape_::kMaxK == 0);
    const bool pad_b1d_head_sz =
        !(param.head_sz % GemvSoftmaxGemvTileShape_::kN1 == 0);

    BOOL_SWITCH_3(
        pad_seqlen, kPadSeqLen, pad_ab0_head_sz, kPadAB0HeadSz, pad_b1d_head_sz,
        kPadB1DHeadSz, [&] {
          using GemvSoftmaxGemvTraits_ =
              ck_tile::GemvSoftmaxGemvTileTraits<kPadSeqLen, kPadAB0HeadSz,
                                                 kPadB1DHeadSz, 1>;

          using GemvSoftmaxGemvPipeline =
              ck_tile::BlockGemvSoftmaxGemvPipeline<GemvSoftmaxGemvProblem_,
                                                    GemvSoftmaxGemvTraits_>;

          using GemvSoftmaxGemvEpilogue =
              ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                  typename GemvSoftmaxGemvTypeConfig<
                      InOutDataType>::GemmAccDataType,
                  InOutDataType, true, kPadB1DHeadSz>>;

          using GemvSoftmaxGemvKernel =
              ck_tile::GemvSoftmaxGemvKernel<GemvSoftmaxGemvPipeline,
                                             GemvSoftmaxGemvEpilogue>;

          RunKernel<GemvSoftmaxGemvKernel>(param, stream);
        });
  };

  template <typename GemvSoftmaxGemvKernel>
  static void RunKernel(const GemvSoftmaxGemvParams& param,
                        hipStream_t stream) {
    const auto kargs = [&] {
      return GemvSoftmaxGemvKernel::MakeKargs(
          param.a_ptr, param.b0_ptr, param.mask_ptr, param.b1_ptr, param.d_ptr,
          param.num_batch, param.seqlen, param.num_head, param.head_sz,
          param.a_batch_stride, param.a_nhead_stride, param.b0_batch_stride,
          param.b0_seq_stride, param.b0_nhead_stride, param.mask_batch_stride,
          param.b1_batch_stride, param.b1_seq_stride, param.b1_nhead_stride,
          param.d_batch_stride, param.d_nhead_stride);
    }();

    dim3 kGridSize = GemvSoftmaxGemvKernel::GridSize(&kargs);
    constexpr dim3 kBlockSize = GemvSoftmaxGemvKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = GemvSoftmaxGemvKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            GemvSoftmaxGemvKernel{}, kGridSize, kBlockSize, 0, kargs));
  };
};

template <typename InOutDataType, typename MaskDataType, ck_tile::index_t kMaxK>
void run_gemv_softmax_gemv(const GemvSoftmaxGemvParams& param,
                           hipStream_t stream) {
  gemv_softmax_gemv_dispatch<InOutDataType, MaskDataType, kMaxK>::Run(param,
                                                                      stream);
};
