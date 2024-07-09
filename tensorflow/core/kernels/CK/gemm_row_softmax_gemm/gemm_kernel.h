// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "gemm_pipeline.h"

// GemmRowSoftmaxGemm problem is solved by two kernels with their pipelines, for each batch:
// S[new_head_sz, seqlen] = A0[new_head_sz, head_sz] @ B0T[seqlen, head_sz]
// B1[new_head_sz, seqlen] = RowSoftmax(S[new_head_sz, seqlen] with filtering by mask
// D[head_sz, seqlen] = A1[head_sz, new_head_sz] @ B1T[seqlen, new_head_sz]

namespace ck_tile {

template <typename GemmPipeline_, typename EpiloguePipeline_>
struct GemmKernel
{
    using GemmPipeline                            = ck_tile::remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = GemmPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = GemmPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using InOutDataType   = ck_tile::remove_cvref_t<typename GemmPipeline::InOutDataType>;
    using GemmAccDataType = ck_tile::remove_cvref_t<typename GemmPipeline::GemmAccDataType>;

    static constexpr bool kPadSeqLen     = GemmPipeline::kPadSeqLen;
    static constexpr bool kPadHeadDim    = GemmPipeline::kPadHeadDim;
    static constexpr bool kPadNewHeadDim = GemmPipeline::kPadNewHeadDim;

    // kargs use aggregate initializer, so no explicit constructor provided
    // user need to use MakeKargs() function to create kargs.
    struct GemmKargs
    {
        const void* b1_ptr;
        const void* a1_ptr;
        void* d_ptr;

        ck_tile::index_t num_batch;
        ck_tile::index_t seqlen;
        ck_tile::index_t b1_head_sz; // new_head_sz
        ck_tile::index_t d_head_sz;  // head_sz
        ck_tile::index_t b1_batch_stride;
        ck_tile::index_t b1_head_stride;
        ck_tile::index_t a1_ld_sz; // leading dim size or stride for the non-leading dim
        ck_tile::index_t d_batch_stride;
        ck_tile::index_t d_head_stride;
    };

    using Kargs = GemmKargs;

    __host__ static constexpr Kargs
    MakeKargs(const void* b1_ptr,
              const void* a1_ptr,
              void* d_ptr,
              ck_tile::index_t num_batch,
              ck_tile::index_t seqlen,
              ck_tile::index_t b1_head_sz, // new_head_sz
              ck_tile::index_t d_head_sz,  // head_sz
              ck_tile::index_t b1_batch_stride,
              ck_tile::index_t b1_head_stride,
              ck_tile::index_t a1_ld_sz, // leading dim size or stride for the non-leading dim
              ck_tile::index_t d_batch_stride,
              ck_tile::index_t d_head_stride)
    {
        Kargs kargs{b1_ptr,
                    a1_ptr,
                    d_ptr,
                    num_batch,
                    seqlen,
                    b1_head_sz,
                    d_head_sz,
                    b1_batch_stride,
                    b1_head_stride,
                    a1_ld_sz,
                    d_batch_stride,
                    d_head_stride};

        return kargs;
    }

    __host__ static constexpr auto GridSize(const Kargs* pKargs)
    {
        return dim3(ck_tile::integer_divide_ceil(pKargs->d_head_sz, GemmPipeline::kM),
                    ck_tile::integer_divide_ceil(pKargs->seqlen, GemmPipeline::kN),
                    pKargs->num_batch);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs* pKargs)
    {
        (void)pKargs;
        ck_tile::index_t i_tile_d_head = blockIdx.x;
        ck_tile::index_t i_tile_seq    = blockIdx.y;
        ck_tile::index_t i_batch       = blockIdx.z;

        return ck_tile::make_tuple(i_batch, i_tile_d_head, i_tile_seq);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return GemmPipeline::GetSmemSize();
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using namespace ck_tile;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide the work-groups
        const auto [i_batch, i_tile_d_head, i_tile_seq] = GetTileIndex(&kargs);

        const index_t i_d_head = __builtin_amdgcn_readfirstlane(i_tile_d_head * GemmPipeline::kM);

        const index_t i_seq = __builtin_amdgcn_readfirstlane(i_tile_seq * GemmPipeline::kN);

        long_index_t batch_offset_b1 = static_cast<long_index_t>(i_batch) * kargs.b1_batch_stride;
        long_index_t batch_offset_d  = static_cast<long_index_t>(i_batch) * kargs.d_batch_stride;

        // for simplicity,  we just prepare the pointer for each batch
        const InOutDataType* b1_ptr = reinterpret_cast<const InOutDataType*>(kargs.b1_ptr) +
                                      batch_offset_b1 + static_cast<long_index_t>(i_seq);
        const InOutDataType* a1_ptr = reinterpret_cast<const InOutDataType*>(kargs.a1_ptr) +
                                      static_cast<long_index_t>(i_d_head) * kargs.a1_ld_sz;

        InOutDataType* d_ptr = reinterpret_cast<InOutDataType*>(kargs.d_ptr) + batch_offset_d +
                               static_cast<long_index_t>(i_d_head) * kargs.d_head_stride +
                               static_cast<long_index_t>(i_seq);

        // B1/A1/D DRAM and DRAM window
        const auto b1t_dram = [&]() {
            const auto b1_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                b1_ptr,
                make_tuple(kargs.b1_head_sz, kargs.seqlen),
                make_tuple(kargs.b1_head_stride, 1),
                number<GemmPipeline::kAlignmentB1>{},
                number<1>{});

            const auto b1t_dram_naive =
                transform_tensor_view(b1_dram_naive,
                                      make_tuple(make_pass_through_transform(kargs.seqlen),
                                                 make_pass_through_transform(kargs.b1_head_sz)),
                                      make_tuple(sequence<1>{}, sequence<0>{}),
                                      make_tuple(sequence<0>{}, sequence<1>{}));

            return pad_tensor_view(
                b1t_dram_naive,
                make_tuple(number<GemmPipeline::kN>{}, number<GemmPipeline::kMaxK>{}),
                sequence<kPadSeqLen, kPadNewHeadDim>{});
        }();

        const auto a1_dram = [&]() {
            const auto a1_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                a1_ptr,
                make_tuple(kargs.d_head_sz, kargs.b1_head_sz), // [head_sz, new_head_sz]
                make_tuple(kargs.a1_ld_sz, 1),
                number<GemmPipeline::kAlignmentA1>{},
                number<1>{});

            return pad_tensor_view(
                a1_dram_naive,
                make_tuple(number<GemmPipeline::kM>{}, number<GemmPipeline::kMaxK>{}),
                sequence<kPadHeadDim, kPadNewHeadDim>{});
        }();

        const auto d_dram = [&]() {
            const auto d_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                d_ptr,
                make_tuple(kargs.d_head_sz, kargs.seqlen),
                make_tuple(kargs.d_head_stride, 1),
                number<GemmPipeline::kAlignmentD>{},
                number<1>{});

            return pad_tensor_view(
                d_dram_naive,
                make_tuple(number<GemmPipeline::kM>{}, number<GemmPipeline::kN>{}),
                sequence<kPadHeadDim, kPadSeqLen>{});
        }();

        auto b1t_dram_window =
            make_tile_window(b1t_dram,
                             make_tuple(number<GemmPipeline::kN>{}, number<GemmPipeline::kMaxK>{}),
                             {0, 0});

        auto a1_dram_window = make_tile_window(
            a1_dram, make_tuple(number<GemmPipeline::kM>{}, number<GemmPipeline::kMaxK>{}), {0, 0});

        auto d_dram_window = make_tile_window(
            d_dram, make_tuple(number<GemmPipeline::kM>{}, number<GemmPipeline::kN>{}), {0, 0});

        auto d_acc_tile = GemmPipeline{}(b1t_dram_window, a1_dram_window, smem_ptr);

        EpiloguePipeline{}(d_dram_window, d_acc_tile);
    }
};

} // namespace ck_tile
