// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "gemm_row_softmax_pipeline.h"

// GemmRowSoftmaxGemm problem is solved by two kernels with their pipelines, for each batch:
// S[new_head_sz, seqlen] = A0[new_head_sz, head_sz] @ B0T[seqlen, head_sz]
// B1[new_head_sz, seqlen] = RowSoftmax(S[new_head_sz, seqlen] with filtering by mask
// D[head_sz, seqlen] = A1[head_sz, new_head_sz] @ B1T[seqlen, new_head_sz]

namespace ck_tile {

template <typename GemmRowSoftmaxPipeline_>
struct GemmRowSoftmaxKernel
{
    using GemmRowSoftmaxPipeline                 = ck_tile::remove_cvref_t<GemmRowSoftmaxPipeline_>;
    static constexpr ck_tile::index_t kBlockSize = GemmRowSoftmaxPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = GemmRowSoftmaxPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using InOutDataType = ck_tile::remove_cvref_t<typename GemmRowSoftmaxPipeline::InOutDataType>;
    using MaskDataType  = ck_tile::remove_cvref_t<typename GemmRowSoftmaxPipeline::MaskDataType>;

    static constexpr bool kPadSeqLen     = GemmRowSoftmaxPipeline::kPadSeqLen;
    static constexpr bool kPadHeadDim    = GemmRowSoftmaxPipeline::kPadHeadDim;
    static constexpr bool kPadNewHeadDim = GemmRowSoftmaxPipeline::kPadNewHeadDim;

    // kargs use aggregate initializer, so no explicit constructor provided,
    // user need to use MakeKargs() function to create kargs.
    struct GemmRowSoftmaxKargs
    {
        const void* b0_ptr;
        const void* a0_ptr;
        const void* mask_ptr;
        void* b1_ptr;

        ck_tile::index_t num_batch;
        ck_tile::index_t seqlen;
        ck_tile::index_t b0_head_sz; // head_sz
        ck_tile::index_t b1_head_sz; // new_head_sz
        ck_tile::index_t b0_batch_stride;
        ck_tile::index_t b0_head_stride;
        ck_tile::index_t a0_ld_sz; // leading dim size or stride for the non-leading dim
        ck_tile::index_t mask_batch_stride;
        ck_tile::index_t b1_batch_stride;
        ck_tile::index_t b1_head_stride;
    };

    using Kargs = GemmRowSoftmaxKargs;

    __host__ static constexpr Kargs
    MakeKargs(const void* b0_ptr,
              const void* a0_ptr,
              const void* mask_ptr,
              void* b1_ptr,
              ck_tile::index_t num_batch,
              ck_tile::index_t seqlen,
              ck_tile::index_t b0_head_sz, // head_sz
              ck_tile::index_t b1_head_sz, // new_head_sz
              ck_tile::index_t b0_batch_stride,
              ck_tile::index_t b0_head_stride,
              ck_tile::index_t a0_ld_sz, // leading dim size or stride for the non-leading dim
              ck_tile::index_t mask_batch_stride,
              ck_tile::index_t b1_batch_stride,
              ck_tile::index_t b1_head_stride)
    {
        Kargs kargs{b0_ptr,
                    a0_ptr,
                    mask_ptr,
                    b1_ptr,
                    num_batch,
                    seqlen,
                    b0_head_sz,
                    b1_head_sz,
                    b0_batch_stride,
                    b0_head_stride,
                    a0_ld_sz,
                    mask_batch_stride,
                    b1_batch_stride,
                    b1_head_stride};

        return kargs;
    }

    __host__ static constexpr auto GridSize(const Kargs* pKargs)
    {
        return dim3(ck_tile::integer_divide_ceil(pKargs->b1_head_sz, GemmRowSoftmaxPipeline::kM),
                    pKargs->num_batch);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs* pKargs)
    {
        (void)pKargs;
        ck_tile::index_t i_tile_b1_head = blockIdx.x;
        ck_tile::index_t i_batch        = blockIdx.y;

        return ck_tile::make_tuple(i_batch, i_tile_b1_head);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return GemmRowSoftmaxPipeline::GetSmemSize();
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using namespace ck_tile;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide the work-groups
        const auto [i_batch, i_tile_b1_head] = GetTileIndex(&kargs);

        const index_t i_b1_head =
            __builtin_amdgcn_readfirstlane(i_tile_b1_head * GemmRowSoftmaxPipeline::kM);

        long_index_t batch_offset_b0 = static_cast<long_index_t>(i_batch) * kargs.b0_batch_stride;
        long_index_t batch_offset_b1 = static_cast<long_index_t>(i_batch) * kargs.b1_batch_stride;
        long_index_t batch_offset_mask =
            static_cast<long_index_t>(i_batch) * kargs.mask_batch_stride;

        // for simplicity, batch just prepare the pointer for each batch
        const InOutDataType* b0_ptr =
            reinterpret_cast<const InOutDataType*>(kargs.b0_ptr) + batch_offset_b0;
        const InOutDataType* a0_ptr = reinterpret_cast<const InOutDataType*>(kargs.a0_ptr) +
                                      static_cast<long_index_t>(i_b1_head) * kargs.a0_ld_sz;

        InOutDataType* b1_ptr = reinterpret_cast<InOutDataType*>(kargs.b1_ptr) + batch_offset_b1 +
                                static_cast<long_index_t>(i_b1_head) * kargs.b1_head_stride;

        const MaskDataType* mask_ptr =
            reinterpret_cast<const MaskDataType*>(kargs.mask_ptr) + batch_offset_mask;

        // B0/A0/B1 DRAM and DRAM window
        const auto b0t_dram = [&]() {
            const auto b0_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                b0_ptr,
                make_tuple(kargs.b0_head_sz, kargs.seqlen),
                make_tuple(kargs.b0_head_stride, 1),
                number<GemmRowSoftmaxPipeline::kAlignmentB0>{},
                number<1>{});

            const auto b0t_dram_naive =
                transform_tensor_view(b0_dram_naive,
                                      make_tuple(make_pass_through_transform(kargs.seqlen),
                                                 make_pass_through_transform(kargs.b0_head_sz)),
                                      make_tuple(sequence<1>{}, sequence<0>{}),
                                      make_tuple(sequence<0>{}, sequence<1>{}));

            return pad_tensor_view(b0t_dram_naive,
                                   make_tuple(number<GemmRowSoftmaxPipeline::kN>{},
                                              number<GemmRowSoftmaxPipeline::kMaxK>{}),
                                   sequence<kPadHeadDim, kPadSeqLen>{});
        }();

        const auto a0_dram = [&]() {
            const auto a0_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                a0_ptr,
                make_tuple(kargs.b1_head_sz, kargs.b0_head_sz), // [new_head_sz, head_sz]
                make_tuple(kargs.a0_ld_sz, 1),
                number<GemmRowSoftmaxPipeline::kAlignmentA0>{},
                number<1>{});

            return pad_tensor_view(a0_dram_naive,
                                   make_tuple(number<GemmRowSoftmaxPipeline::kM>{},
                                              number<GemmRowSoftmaxPipeline::kMaxK>{}),
                                   sequence<kPadNewHeadDim, kPadHeadDim>{});
        }();

        // extend mask from [seqlen] to [kM, seqlen]
        const auto mask_dram = [&]() {
            const auto mask_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                mask_ptr,
                make_tuple(kargs.seqlen),
                make_tuple(1),
                number<GemmRowSoftmaxPipeline::kAlignmentMask>{},
                number<1>{});

            return pad_tensor_view(mask_dram_naive,
                                   make_tuple(number<GemmRowSoftmaxPipeline::kN>{}),
                                   sequence<kPadSeqLen>{});
        }();

        const auto b1_dram = [&]() {
            const auto b1_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                b1_ptr,
                make_tuple(kargs.b1_head_sz, kargs.seqlen),
                make_tuple(kargs.b1_head_stride, 1),
                number<GemmRowSoftmaxPipeline::kAlignmentB1>{},
                number<1>{});

            return pad_tensor_view(b1_dram_naive,
                                   make_tuple(number<GemmRowSoftmaxPipeline::kM>{},
                                              number<GemmRowSoftmaxPipeline::kN>{}),
                                   sequence<kPadNewHeadDim, kPadSeqLen>{});
        }();

        auto b0t_dram_window = make_tile_window(b0t_dram,
                                                make_tuple(number<GemmRowSoftmaxPipeline::kN>{},
                                                           number<GemmRowSoftmaxPipeline::kMaxK>{}),
                                                {0, 0});

        auto a0_dram_window = make_tile_window(a0_dram,
                                               make_tuple(number<GemmRowSoftmaxPipeline::kM>{},
                                                          number<GemmRowSoftmaxPipeline::kMaxK>{}),
                                               {0, 0});

        auto mask_dram_window =
            make_tile_window(mask_dram, make_tuple(number<GemmRowSoftmaxPipeline::kN>{}), {0});

        auto b1_dram_window = make_tile_window(
            b1_dram,
            make_tuple(number<GemmRowSoftmaxPipeline::kM>{}, number<GemmRowSoftmaxPipeline::kN>{}),
            {0, 0});

        GemmRowSoftmaxPipeline{}(b0t_dram_window,
                                 a0_dram_window,
                                 mask_dram_window,
                                 b1_dram_window,
                                 kargs.seqlen,
                                 smem_ptr);

        // no epilogue is used, since b1 tiles are written inside the pipeline loop
    }
};

} // namespace ck_tile
