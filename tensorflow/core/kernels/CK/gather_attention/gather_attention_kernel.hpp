// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "gather_attention_pipeline.hpp"

// clang-format off
// GatherAttention problem is solved by one kernel, which takes A/B0/B1/mask tensor and indices array as input, do the following:
// S[num_batch, num_head, seqlen] = A[num_batch, num_head, head_sz] @ BOT[num_batch, seqlen, num_head, head_sz]
// P[num_head, num_batch, seqlen] = RowSoftmax(S[num_batch, num_head, seqlen] with filtering by mask before softmax-ing
// D[num_batch, num_head, head_sz] = P[num_head, num_batch, seqlen] @ B1T[num_batch, seqlen, num_head, head_sz]
// The process is very similar to the decoder attention with one Query token in each batch. But one specical point of
// GatherAttention is that the accessed batches of B0/B1/mask tensors are specified by an indices vector.
// clang-format on

namespace ck_tile {

template <typename GatherAttentionPipeline_, typename EpiloguePipeline_>
struct GatherAttentionKernel {
  using GatherAttentionPipeline =
      ck_tile::remove_cvref_t<GatherAttentionPipeline_>;
  using EpiloguePipeline = ck_tile::remove_cvref_t<EpiloguePipeline_>;
  static constexpr ck_tile::index_t kBlockSize =
      GatherAttentionPipeline::kBlockSize;
  static constexpr ck_tile::index_t kBlockPerCu =
      GatherAttentionPipeline::kBlockPerCu;
  static_assert(kBlockPerCu > 0);

  using InOutDataType =
      ck_tile::remove_cvref_t<typename GatherAttentionPipeline::InOutDataType>;
  using MaskDataType =
      ck_tile::remove_cvref_t<typename GatherAttentionPipeline::MaskDataType>;

  static constexpr bool kPadSeqLen = GatherAttentionPipeline::kPadSeqLen;
  static constexpr bool kPadAB0HeadSz = GatherAttentionPipeline::kPadAB0HeadSz;
  static constexpr bool kPadB1DHeadSz = GatherAttentionPipeline::kPadB1DHeadSz;

  // kargs use aggregate initializer, so no explicit constructor provided,
  // user need to use MakeKargs() function to create kargs.
  struct GatherAttentionKargs {
    const void* a_ptr;
    const void* b0_ptr;
    const void* mask_ptr;
    const int* indices_ptr;
    const void* b1_ptr;
    void* d_ptr;

    ck_tile::index_t num_index;
    ck_tile::index_t seqlen;
    ck_tile::index_t num_head;
    ck_tile::index_t head_sz;
    ck_tile::index_t a_batch_stride;
    ck_tile::index_t a_nhead_stride;
    ck_tile::index_t b0_batch_stride;
    ck_tile::index_t b0_seq_stride;
    ck_tile::index_t b0_nhead_stride;
    ck_tile::index_t mask_batch_stride;
    ck_tile::index_t b1_batch_stride;
    ck_tile::index_t b1_seq_stride;
    ck_tile::index_t b1_nhead_stride;
    ck_tile::index_t d_batch_stride;
    ck_tile::index_t d_nhead_stride;
  };

  using Kargs = GatherAttentionKargs;

  __host__ static constexpr Kargs MakeKargs(
      const void* a_ptr, const void* b0_ptr, const void* mask_ptr,
      const void* indices_ptr, const void* b1_ptr, void* d_ptr,
      ck_tile::index_t num_index, ck_tile::index_t seqlen,
      ck_tile::index_t num_head, ck_tile::index_t head_sz,
      ck_tile::index_t a_batch_stride, ck_tile::index_t a_nhead_stride,
      ck_tile::index_t b0_batch_stride, ck_tile::index_t b0_seq_stride,
      ck_tile::index_t b0_nhead_stride, ck_tile::index_t mask_batch_stride,
      ck_tile::index_t b1_batch_stride, ck_tile::index_t b1_seq_stride,
      ck_tile::index_t b1_nhead_stride, ck_tile::index_t d_batch_stride,
      ck_tile::index_t d_nhead_stride) {
    Kargs kargs{a_ptr,           b0_ptr,
                mask_ptr,        reinterpret_cast<const int*>(indices_ptr),
                b1_ptr,          d_ptr,
                num_index,       seqlen,
                num_head,        head_sz,
                a_batch_stride,  a_nhead_stride,
                b0_batch_stride, b0_seq_stride,
                b0_nhead_stride, mask_batch_stride,
                b1_batch_stride, b1_seq_stride,
                b1_nhead_stride, d_batch_stride,
                d_nhead_stride};

    return kargs;
  }

  __host__ static constexpr auto GridSize(const Kargs* pKargs) {
    return dim3(pKargs->num_head, pKargs->num_index);
  }

  CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs* pKargs) {
    (void)pKargs;
    ck_tile::index_t i_nhead = blockIdx.x;
    ck_tile::index_t i_batch = blockIdx.y;

    return ck_tile::make_tuple(i_batch, i_nhead);
  }

  __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
    return GatherAttentionPipeline::GetSmemSize();
  }

  CK_TILE_DEVICE void operator()(Kargs kargs) const {
    using namespace ck_tile;

    // allocate LDS
    __shared__ char smem_ptr[GetSmemSize()];

    // divide the work-groups
    const auto [i_batch, i_nhead] = GetTileIndex(&kargs);
    const auto i_batch_phy = kargs.indices_ptr[i_batch];

    long_index_t block_offset_a =
        static_cast<long_index_t>(i_batch) * kargs.a_batch_stride +
        static_cast<long_index_t>(i_nhead) * kargs.a_nhead_stride;
    long_index_t block_offset_b0 =
        static_cast<long_index_t>(i_batch_phy) * kargs.b0_batch_stride +
        static_cast<long_index_t>(i_nhead) * kargs.b0_nhead_stride;
    long_index_t block_offset_mask =
        static_cast<long_index_t>(i_batch_phy) * kargs.mask_batch_stride;
    long_index_t block_offset_b1 =
        static_cast<long_index_t>(i_batch_phy) * kargs.b1_batch_stride +
        static_cast<long_index_t>(i_nhead) * kargs.b1_nhead_stride;
    long_index_t block_offset_d =
        static_cast<long_index_t>(i_batch) * kargs.d_batch_stride +
        static_cast<long_index_t>(i_nhead) * kargs.d_nhead_stride;

    const InOutDataType* a_ptr =
        reinterpret_cast<const InOutDataType*>(kargs.a_ptr) + block_offset_a;
    const InOutDataType* b0_ptr =
        reinterpret_cast<const InOutDataType*>(kargs.b0_ptr) + block_offset_b0;
    const InOutDataType* mask_ptr =
        reinterpret_cast<const MaskDataType*>(kargs.mask_ptr) +
        block_offset_mask;
    const InOutDataType* b1_ptr =
        reinterpret_cast<const InOutDataType*>(kargs.b1_ptr) + block_offset_b1;
    InOutDataType* d_ptr =
        reinterpret_cast<InOutDataType*>(kargs.d_ptr) + block_offset_d;

    // A0/B0/B1/D/Mask DRAM and DRAM window
    const auto a_dram = [&]() {
      const auto a_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              a_ptr, make_tuple(1, kargs.head_sz), make_tuple(kargs.head_sz, 1),
              number<GatherAttentionPipeline::kAlignmentA>{}, number<1>{});

      return pad_tensor_view(
          a_dram_naive,
          make_tuple(number<GatherAttentionPipeline::kM>{},
                     number<GatherAttentionPipeline::kMaxK>{}),
          sequence<true, kPadAB0HeadSz>{});
    }();

    const auto b0_dram = [&]() {
      const auto b0_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              b0_ptr, make_tuple(kargs.seqlen, kargs.head_sz),
              make_tuple(kargs.b0_seq_stride, 1),
              number<GatherAttentionPipeline::kAlignmentB0>{}, number<1>{});

      return pad_tensor_view(b0_dram_naive,
                             make_tuple(number<GatherAttentionPipeline::kN0>{},
                                        number<GatherAttentionPipeline::kK0>{}),
                             sequence<kPadSeqLen, kPadAB0HeadSz>{});
    }();

    const auto b1t_dram = [&]() {
      const auto b1_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              b1_ptr, make_tuple(kargs.seqlen, kargs.head_sz),
              make_tuple(kargs.b1_seq_stride, 1),
              number<GatherAttentionPipeline::kAlignmentB1>{}, number<1>{});

      const auto b1t_dram_naive = transform_tensor_view(
          b1_dram_naive,
          make_tuple(make_pass_through_transform(kargs.head_sz),
                     make_pass_through_transform(kargs.seqlen)),
          make_tuple(sequence<1>{}, sequence<0>{}),
          make_tuple(sequence<0>{}, sequence<1>{}));

      return pad_tensor_view(b1t_dram_naive,
                             make_tuple(number<GatherAttentionPipeline::kN1>{},
                                        number<GatherAttentionPipeline::kK1>{}),
                             sequence<kPadB1DHeadSz, kPadSeqLen>{});
    }();

    const auto mask_dram = [&]() {
      const auto mask_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              mask_ptr, make_tuple(kargs.seqlen), make_tuple(1),
              number<GatherAttentionPipeline::kAlignmentMask>{}, number<1>{});

      return pad_tensor_view(mask_dram_naive,
                             make_tuple(number<GatherAttentionPipeline::kN0>{}),
                             sequence<kPadSeqLen>{});
    }();

    // pad the rows from 1 to kM, actual outputting will be only one row
    const auto d_dram = [&]() {
      const auto d_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              d_ptr, make_tuple(1, kargs.head_sz), make_tuple(kargs.head_sz, 1),
              number<GatherAttentionPipeline::kAlignmentD>{}, number<1>{});

      return pad_tensor_view(d_dram_naive,
                             make_tuple(number<GatherAttentionPipeline::kM>{},
                                        number<GatherAttentionPipeline::kN1>{}),
                             sequence<true, kPadB1DHeadSz>{});
    }();

    auto a_dram_window =
        make_tile_window(a_dram,
                         make_tuple(number<GatherAttentionPipeline::kM>{},
                                    number<GatherAttentionPipeline::kMaxK>{}),
                         {0, 0});

    auto b0_dram_window =
        make_tile_window(b0_dram,
                         make_tuple(number<GatherAttentionPipeline::kN0>{},
                                    number<GatherAttentionPipeline::kK0>{}),
                         {0, 0});

    auto b1t_dram_window =
        make_tile_window(b1t_dram,
                         make_tuple(number<GatherAttentionPipeline::kN1>{},
                                    number<GatherAttentionPipeline::kK1>{}),
                         {0, 0});

    auto mask_dram_window = make_tile_window(
        mask_dram, make_tuple(number<GatherAttentionPipeline::kN0>{}), {0});

    auto d_dram_window =
        make_tile_window(d_dram,
                         make_tuple(number<GatherAttentionPipeline::kM>{},
                                    number<GatherAttentionPipeline::kN1>{}),
                         {0, 0});

    auto d_acc_tile = GatherAttentionPipeline{}(
        a_dram_window, b0_dram_window, mask_dram_window, b1t_dram_window,
        kargs.seqlen, kargs.head_sz, smem_ptr);

    EpiloguePipeline{}(d_dram_window, d_acc_tile);
  }
};

}  // namespace ck_tile
