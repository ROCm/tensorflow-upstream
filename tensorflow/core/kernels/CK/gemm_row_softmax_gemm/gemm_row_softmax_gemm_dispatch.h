// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "gemm_row_softmax_gemm_problem.h"
#include "gemm_row_softmax_gemm_tile_traits.h"
#include "gemm_row_softmax_pipeline.h"
#include "gemm_pipeline.h"
#include "gemm_row_softmax_kernel.h"
#include "gemm_kernel.h"

#include "gemm_row_softmax_gemm_setting.h"
#include "gemm_row_softmax_gemm_params.h"
#include "bool_switch.h"

template <typename InOutDataType,
          typename MaskDataType,
          ck_tile::index_t Gemm0MaxK,
          ck_tile::index_t Gemm1MaxK>
struct gemm_row_softmax_gemm_dispatch
{
    using GemmRSGemmProblem_ = ck_tile::GemmRowSoftmaxGemmProblem<
        InOutDataType,
        typename GemmRowSoftmaxGemmTypeConfig<InOutDataType>::GemmAccDataType,
        typename GemmRowSoftmaxGemmTypeConfig<InOutDataType>::SMComputeDataType,
        MaskDataType,
        Gemm0RowSoftmaxTileShape<Gemm0MaxK>,
        Gemm1TileShape<Gemm1MaxK>>;

    static void Run(const GemmRowSoftmaxGemmParams& param, hipStream_t stream)
    {
        {
            using Gemm0RowSoftmaxTileShape_ = Gemm0RowSoftmaxTileShape<Gemm0MaxK>;

            const bool pad_seqlen   = !(param.seqlen % Gemm0RowSoftmaxTileShape_::kGemmN == 0);
            const bool pad_head     = !(param.b0_head_sz % Gemm0RowSoftmaxTileShape_::kMaxK == 0);
            const bool pad_new_head = !(param.b1_head_sz % Gemm0RowSoftmaxTileShape_::kGemmM == 0);

            BOOL_SWITCH_3(
                pad_seqlen, kPadSeqLen, pad_head, kPadHeadDim, pad_new_head, kPadNewHeadDim, [&] {
                    using Gemm0RowSoftmaxTraits_ =
                        ck_tile::GemmTileTraits<kPadNewHeadDim, kPadSeqLen, kPadHeadDim, 2>;

                    using Gemm0RowSoftmaxPipeline =
                        ck_tile::BlockGemmRowSoftmaxPipeline<GemmRSGemmProblem_,
                                                             Gemm0RowSoftmaxTraits_>;

                    using Gemm0RowSoftmaxKernel =
                        ck_tile::GemmRowSoftmaxKernel<Gemm0RowSoftmaxPipeline>;

                    RunWithGemmRowSoftmaxKernel<Gemm0RowSoftmaxKernel>(param, stream);
                });
        };

        {
            using Gemm1TileShape_ = Gemm1TileShape<Gemm1MaxK>;

            const bool pad_seqlen   = !(param.seqlen % Gemm1TileShape_::kGemmN == 0);
            const bool pad_head     = !(param.b0_head_sz % Gemm1TileShape_::kGemmM == 0);
            const bool pad_new_head = !(param.b1_head_sz % Gemm1TileShape_::kMaxK == 0);

            BOOL_SWITCH_3(
                pad_seqlen, kPadSeqLen, pad_head, kPadHeadDim, pad_new_head, kPadNewHeadDim, [&] {
                    using Gemm1Traits_ =
                        ck_tile::GemmTileTraits<kPadHeadDim, kPadSeqLen, kPadNewHeadDim, 2>;

                    using Gemm1Pipeline =
                        ck_tile::BlockGemmPipeline<GemmRSGemmProblem_, Gemm1Traits_>;

                    using Gemm1Epilogue =
                        ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                            typename GemmRowSoftmaxGemmTypeConfig<InOutDataType>::GemmAccDataType,
                            InOutDataType,
                            kPadHeadDim,
                            kPadSeqLen>>;

                    using Gemm1Kernel = ck_tile::GemmKernel<Gemm1Pipeline, Gemm1Epilogue>;

                    RunWithGemmKernel<Gemm1Kernel>(param, stream);
                });
        };
    };

    template <typename GemmRowSoftmaxKernel>
    static void RunWithGemmRowSoftmaxKernel(const GemmRowSoftmaxGemmParams& param,
                                            hipStream_t stream)
    {
        const auto kargs = [&] {
            return GemmRowSoftmaxKernel::MakeKargs(param.b0_ptr,
                                                   param.a0_ptr,
                                                   param.mask_ptr,
                                                   param.b1_ptr,
                                                   param.num_batch,
                                                   param.seqlen,
                                                   param.b0_head_sz,
                                                   param.b1_head_sz,
                                                   param.b0_batch_stride,
                                                   param.b0_head_stride,
                                                   param.a0_ld_sz,
                                                   param.mask_batch_stride,
                                                   param.b1_batch_stride,
                                                   param.b1_head_stride);
        }();

        dim3 kGridSize                         = GemmRowSoftmaxKernel::GridSize(&kargs);
        constexpr dim3 kBlockSize              = GemmRowSoftmaxKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = GemmRowSoftmaxKernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                                     ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
                                         GemmRowSoftmaxKernel{}, kGridSize, kBlockSize, 0, kargs));
    };

    template <typename GemmKernel>
    static void RunWithGemmKernel(const GemmRowSoftmaxGemmParams& param, hipStream_t stream)
    {
        const auto kargs = [&] {
            return GemmKernel::MakeKargs(param.b1_ptr,
                                         param.a1_ptr,
                                         param.d_ptr,
                                         param.num_batch,
                                         param.seqlen,
                                         param.b1_head_sz,
                                         param.b0_head_sz,
                                         param.b1_batch_stride,
                                         param.b1_head_stride,
                                         param.a1_ld_sz,
                                         param.d_batch_stride,
                                         param.d_head_stride);
        }();

        dim3 kGridSize                         = GemmKernel::GridSize(&kargs);
        constexpr dim3 kBlockSize              = GemmKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = GemmKernel::kBlockPerCu;

        (void)ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                                     ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
                                         GemmKernel{}, kGridSize, kBlockSize, 0, kargs));
    };
};

template <typename InOutDataType,
          typename MaskDataType,
          ck_tile::index_t Gemm0MaxK,
          ck_tile::index_t Gemm1MaxK>
void run_gemm_row_softmax_gemm(const GemmRowSoftmaxGemmParams& param, hipStream_t stream)
{
    gemm_row_softmax_gemm_dispatch<InOutDataType, MaskDataType, Gemm0MaxK, Gemm1MaxK>::Run(param,
                                                                                           stream);
};
