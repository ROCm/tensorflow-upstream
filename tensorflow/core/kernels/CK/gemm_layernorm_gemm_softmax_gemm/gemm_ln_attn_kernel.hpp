// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename TilePartitioner_, typename GemmLnAttnPipeline_, typename EpiloguePipeline_>
struct GemmLnAttnKernel
{
    using TilePartitioner                         = ck_tile::remove_cvref_t<TilePartitioner_>;
    using GemmLnAttnPipeline                      = ck_tile::remove_cvref_t<GemmLnAttnPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = GemmLnAttnPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = GemmLnAttnPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = GemmLnAttnPipeline::Problem::kBlockPerCu;

    using QDataType       = ck_tile::remove_cvref_t<typename GemmLnAttnPipeline::QDataType>;
    using WDataType       = ck_tile::remove_cvref_t<typename GemmLnAttnPipeline::WDataType>;
    using BiasDataType    = ck_tile::remove_cvref_t<typename GemmLnAttnPipeline::BiasDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename GemmLnAttnPipeline::GammaDataType>;
    using BetaDataType    = ck_tile::remove_cvref_t<typename GemmLnAttnPipeline::BetaDataType>;
    using KDataType       = ck_tile::remove_cvref_t<typename GemmLnAttnPipeline::KDataType>;
    using KeyMaskDataType = ck_tile::remove_cvref_t<typename GemmLnAttnPipeline::KeyMaskDataType>;
    using VDataType       = ck_tile::remove_cvref_t<typename GemmLnAttnPipeline::VDataType>;
    using ODataType       = ck_tile::remove_cvref_t<typename GemmLnAttnPipeline::ODataType>;

    static constexpr bool kPadBatchQ   = GemmLnAttnPipeline::kPadBatchQ;
    static constexpr bool kPadQWGemmK  = GemmLnAttnPipeline::kPadQWGemmK;
    static constexpr bool kPadSeqLenK  = GemmLnAttnPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = GemmLnAttnPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = GemmLnAttnPipeline::kPadHeadDimV;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    // clang-format on

    __host__ static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        using bfs = typename GemmLnAttnPipeline::BlockGemmLnAttnShape;
        using gbr = typename bfs::Gemm0BlockWarps;
        using gwt = typename bfs::Gemm0WarpTile;
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadBatchQ) n += "sq";
            if (kPadQWGemmK) n += "w";
            if (kPadSeqLenK) n += "sk";
            if (kPadHeadDimQ) n += "d";
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("gemm_ln_attn_") + _SS_(t2s<QDataType>::name) +
            "_" + _SS_(TilePartitioner::name) + "_" +
            "b" + _TS_(bfs::kM0) + "x" + _TS_(bfs::kN0) + "x" + _TS_(bfs::kK0) + "_" +
                    _TS_(bfs::kN1) + "x" + _TS_(bfs::kK1) + "_" +
                    _TS_(bfs::kN2) + "x" + _TS_(bfs::kK2) + "_" +
            "r" + _TS_(gbr::at(ck_tile::number<0>{})) + "x" + _TS_(gbr::at(ck_tile::number<1>{})) + "x" + _TS_(gbr::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(gwt::at(ck_tile::number<0>{})) + "x" + _TS_(gwt::at(ck_tile::number<1>{})) + "x" + _TS_(gwt::at(ck_tile::number<2>{})) + "_" +
            (kBlockPerCuInput == -1 ? "" : ("o" + _TS_(kBlockPerCu) + "_")) + _SS_(GemmLnAttnPipeline::name) + "_" + (pn.empty() ? "" : "_" + pn);
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct GemmLnAttnEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct GemmLnAttnCommonKargs
    {
        const void* q_ptr;
        const void* w_ptr;
        const void* bias_ptr;
        const void* gamma_ptr;
        const void* beta_ptr;
        const void* k_ptr;
        const void* keymask_ptr;
        const void* v_ptr;
        void* o_ptr;

        ck_tile::index_t batch_q;
        ck_tile::index_t qw_k;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t batch_kv;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;
        ck_tile::index_t num_head;
        float scale_s;
        float lrelu_alpha;
        bool do_leaky_relu;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_w;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_w;
        ck_tile::index_t nhead_stride_bias;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o;

        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_keymask;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;
    };

    using Kargs = GemmLnAttnCommonKargs;

    __host__ static constexpr Kargs MakeKargs(const void* q_ptr,
                                              const void* w_ptr,
                                              const void* bias_ptr,
                                              const void* k_ptr,
                                              const void* v_ptr,
                                              const void* gamma_ptr,
                                              const void* beta_ptr,
                                              const void* keymask_ptr,
                                              void* o_ptr,
                                              ck_tile::index_t K,
                                              ck_tile::index_t M,
                                              ck_tile::index_t N0,
                                              ck_tile::index_t N1,
                                              ck_tile::index_t N2,
                                              ck_tile::index_t batch_kv,
                                              ck_tile::index_t nhead,
                                              float lrelu_alpha,
                                              bool do_leaky_relu)
    {
        const ck_tile::index_t matw_k   = K;
        const ck_tile::index_t batch_q  = M;
        const ck_tile::index_t hdim_q   = N0 / nhead;
        const ck_tile::index_t seqlen_k = N1;
        const ck_tile::index_t hdim_v   = N2 / nhead;
        const float scale_s             = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q));

        // setup intra-block stride_* arguments for 2D+ tensors
        /* bias, gamma, beta and keymask are 1D tensors in block level, stride equal to 1 */
        const ck_tile::index_t stride_q = matw_k;
        const ck_tile::index_t stride_w = matw_k;
        const ck_tile::index_t stride_k = nhead * hdim_q;
        const ck_tile::index_t stride_v = nhead * hdim_v;
        const ck_tile::index_t stride_o = nhead * hdim_v;

        // setup inter-block offset on nhead dim, nhead_stride_* arguments
        /* q, gamma, beta and keymask don't have nhead included in dimension, stride equal to 0 */
        const ck_tile::index_t nhead_stride_w    = hdim_q * matw_k;
        const ck_tile::index_t nhead_stride_bias = hdim_q;
        const ck_tile::index_t nhead_stride_k    = hdim_q;
        const ck_tile::index_t nhead_stride_v    = hdim_v;
        const ck_tile::index_t nhead_stride_o    = hdim_v;

        // setup inter-block offset on batch_kv dim, batch_stride_* arguments
        /* q, w, bias, gamma, beta don't have batch_kv included in dimension, stride equal to 0 */
        const ck_tile::index_t batch_stride_k       = seqlen_k * nhead * hdim_q;
        const ck_tile::index_t batch_stride_keymask = seqlen_k;
        const ck_tile::index_t batch_stride_v       = seqlen_k * nhead * hdim_v;
        const ck_tile::index_t batch_stride_o       = batch_q * nhead * hdim_v;

        Kargs kargs{q_ptr,
                    w_ptr,
                    bias_ptr,
                    gamma_ptr,
                    beta_ptr,
                    k_ptr,
                    keymask_ptr,
                    v_ptr,
                    o_ptr,
                    batch_q,
                    matw_k,
                    seqlen_k,
                    batch_kv,
                    hdim_q,
                    hdim_v,
                    nhead,
                    static_cast<float>(scale_s * ck_tile::log2e_v<>),
                    lrelu_alpha,
                    do_leaky_relu,
                    stride_q,
                    stride_w,
                    stride_k,
                    stride_v,
                    stride_o,
                    nhead_stride_w,
                    nhead_stride_bias,
                    nhead_stride_k,
                    nhead_stride_v,
                    nhead_stride_o,
                    batch_stride_k,
                    batch_stride_keymask,
                    batch_stride_v,
                    batch_stride_o};

        return kargs;
    }

    __host__ static constexpr auto
    GridSize(ck_tile::index_t batch_kv, ck_tile::index_t nhead, ck_tile::index_t batch_q)
    {
        return TilePartitioner::GridSize(batch_kv, nhead, batch_q);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(GemmLnAttnPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = TilePartitioner{}(kargs.batch_q);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * GemmLnAttnPipeline::kM0);

        // Global buffer:
        // Stage1: Q, W, Bias, Gamma, Beta
        /* Don't have batch_kv introduced offset */
        // Stage2: K, KeyMask, V
        long_index_t batch_offset_k       = 0;
        long_index_t batch_offset_keymask = 0;
        long_index_t batch_offset_v       = 0;
        long_index_t batch_offset_o       = 0;

        batch_offset_k       = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
        batch_offset_keymask = static_cast<long_index_t>(i_batch) * kargs.batch_stride_keymask;
        batch_offset_v       = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
        batch_offset_o       = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr);

        const WDataType* w_ptr = reinterpret_cast<const WDataType*>(kargs.w_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_w;

        const BiasDataType* bias_ptr = reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                                       static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_bias;

        // Per-Head LayerNorm, don't have head-wise offset
        const GammaDataType* gamma_ptr = reinterpret_cast<const GammaDataType*>(kargs.gamma_ptr);

        const BetaDataType* beta_ptr = reinterpret_cast<const BetaDataType*>(kargs.beta_ptr);

        const KDataType* k_ptr = reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_k +
                                 batch_offset_k;

        // KeyMask was applied to 2nd GemmC, don't have head-related dimension
        const KeyMaskDataType* keymask_ptr =
            reinterpret_cast<const KeyMaskDataType*>(kargs.keymask_ptr) + batch_offset_keymask;

        const VDataType* v_ptr = reinterpret_cast<const VDataType*>(kargs.v_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_v +
                                 batch_offset_v;

        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // DRAM
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.batch_q, kargs.qw_k),
                make_tuple(kargs.stride_q, 1),
                number<GemmLnAttnPipeline::kAlignmentQ>{},
                number<1>{});

            return pad_tensor_view(
                q_dram_naive,
                make_tuple(number<GemmLnAttnPipeline::kM0>{}, number<GemmLnAttnPipeline::kK0>{}),
                sequence<kPadBatchQ, kPadQWGemmK>{});
        }();

        const auto w_dram = [&]() {
            const auto w_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                w_ptr,
                make_tuple(kargs.hdim_q, kargs.qw_k),
                make_tuple(kargs.stride_w, 1),
                number<GemmLnAttnPipeline::kAlignmentW>{},
                number<1>{});

            return pad_tensor_view(
                w_dram_naive,
                make_tuple(number<GemmLnAttnPipeline::kN0>{}, number<GemmLnAttnPipeline::kK0>{}),
                sequence<kPadHeadDimQ, kPadQWGemmK>{});
        }();

        const auto bias_dram = [&]() {
            const auto bias_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                bias_ptr,
                make_tuple(kargs.hdim_q),
                make_tuple(1),
                number<GemmLnAttnPipeline::kAlignmentBias>{},
                number<1>{});

            return pad_tensor_view(bias_dram_naive,
                                   make_tuple(number<GemmLnAttnPipeline::kN0>{}),
                                   sequence<kPadHeadDimQ>{});
        }();

        const auto gamma_dram = [&]() {
            const auto gamma_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                gamma_ptr,
                make_tuple(kargs.hdim_q),
                make_tuple(1),
                number<GemmLnAttnPipeline::kAlignmentGamma>{},
                number<1>{});

            return pad_tensor_view(gamma_dram_naive,
                                   make_tuple(number<GemmLnAttnPipeline::kN0>{}),
                                   sequence<kPadHeadDimQ>{});
        }();

        const auto beta_dram = [&]() {
            const auto beta_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                beta_ptr,
                make_tuple(kargs.hdim_q),
                make_tuple(1),
                number<GemmLnAttnPipeline::kAlignmentBeta>{},
                number<1>{});

            return pad_tensor_view(beta_dram_naive,
                                   make_tuple(number<GemmLnAttnPipeline::kN0>{}),
                                   sequence<kPadHeadDimQ>{});
        }();

        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<GemmLnAttnPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<GemmLnAttnPipeline::kN1>{}, number<GemmLnAttnPipeline::kK1>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();

        const auto keymask_dram = [&]() {
            const auto keymask_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                keymask_ptr,
                make_tuple(kargs.seqlen_k),
                make_tuple(1),
                number<GemmLnAttnPipeline::kAlignmentKeyMask>{},
                number<1>{});

            return pad_tensor_view(keymask_dram_naive,
                                   make_tuple(number<GemmLnAttnPipeline::kN1>{}),
                                   sequence<kPadHeadDimQ>{});
        }();

        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                number<GemmLnAttnPipeline::kAlignmentV>{},
                number<1>{});

            return pad_tensor_view(
                v_dram_naive,
                make_tuple(number<GemmLnAttnPipeline::kK2>{}, number<GemmLnAttnPipeline::kN2>{}),
                sequence<kPadSeqLenK, kPadHeadDimV>{});
        }();

        // DRAM window
        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<GemmLnAttnPipeline::kM0>{}, number<GemmLnAttnPipeline::kK0>{}),
            {i_m0, 0});

        auto w_dram_window = make_tile_window(
            w_dram,
            make_tuple(number<GemmLnAttnPipeline::kN0>{}, number<GemmLnAttnPipeline::kK0>{}),
            {0, 0});

        auto bias_dram_window =
            make_tile_window(bias_dram, make_tuple(number<GemmLnAttnPipeline::kN0>{}), {0});

        auto gamma_dram_window =
            make_tile_window(gamma_dram, make_tuple(number<GemmLnAttnPipeline::kN0>{}), {0});

        auto beta_dram_window =
            make_tile_window(beta_dram, make_tuple(number<GemmLnAttnPipeline::kN0>{}), {0});

        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<GemmLnAttnPipeline::kN1>{}, number<GemmLnAttnPipeline::kK1>{}),
            {0, 0});

        auto keymask_dram_window =
            make_tile_window(keymask_dram, make_tuple(number<GemmLnAttnPipeline::kN1>{}), {0});

        auto v_dram_window = make_tile_window(
            v_dram,
            make_tuple(number<GemmLnAttnPipeline::kK2>{}, number<GemmLnAttnPipeline::kN2>{}),
            {0, 0});

        auto o_acc_tile = GemmLnAttnPipeline{}(q_dram_window,
                                               w_dram_window,
                                               bias_dram_window,
                                               gamma_dram_window,
                                               beta_dram_window,
                                               k_dram_window,
                                               keymask_dram_window,
                                               v_dram_window,
                                               kargs.qw_k,
                                               kargs.hdim_q,
                                               kargs.seqlen_k,
                                               kargs.scale_s,
                                               kargs.lrelu_alpha,
                                               kargs.do_leaky_relu,
                                               smem_ptr);

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.batch_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<GemmLnAttnPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<GemmLnAttnPipeline::kM0>{}, number<GemmLnAttnPipeline::kN2>{}),
                sequence<kPadBatchQ, kPadHeadDimV>{});
        }();

        auto o_dram_window = make_tile_window(
            o_dram,
            make_tuple(number<GemmLnAttnPipeline::kM0>{}, number<GemmLnAttnPipeline::kN2>{}),
            {i_m0, 0});

        EpiloguePipeline{}(o_dram_window, o_acc_tile);
    }
};

} // namespace ck_tile