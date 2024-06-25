#pragma once
// macros for cutlass changes required by ali customized logic, must be defined before cutlass includes

#include "cuda_runtime.h"
#endif
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/cutlass.h"
#include "auto_gen/device/FusedGemmBiasAdd.h"
#include "tensorflow/core/kernels/cutlass/include/arch_type.h"
template<int N_TILE>
void FusedGemmBiasAdd_turing_impl(
    int M,
    int N,
    int K,
    int Batch,
    void* A0,
    void* B0,
    void* C0,
    void* D0,
    STREAM_TYPE stream
){
using Gemm0 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<32, N_TILE, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, CUTLASS_INSTRUCTION_N_SIZE, CUTLASS_INSTRUCTION_K_SIZE>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    8,
    8>;


cutlass::half_t alpha0 = cutlass::half_t(1);
cutlass::half_t beta0 = cutlass::half_t(1);
cutlass::gemm::GemmCoord problem_size_0(M, N, K);
typename Gemm0::Arguments arguments_0{
    problem_size_0,
    {reinterpret_cast<cutlass::half_t*>(A0), K}, M * K,
    {reinterpret_cast<cutlass::half_t*>(B0), K}, N * K,
    {reinterpret_cast<cutlass::half_t*>(C0), 0}, N,
    {reinterpret_cast<cutlass::half_t*>(D0), N}, M * N,
    { alpha0, beta0 },
    Batch};
    Gemm0 gemm_op_0;
    gemm_op_0.initialize(arguments_0, nullptr);

    gemm_op_0(STREAM_CAST(stream));

}

template<int N_TILE>
void FusedGemmBiasAdd_volta_impl(
    int M,
    int N,
    int K,
    int Batch,
    void* A0,
    void* B0,
    void* C0,
    void* D0,
STREAM_TYPE stream){
using Gemm0 = cutlass::gemm::device::GemmBatched<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<32, N_TILE, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2,
    8,
    8>;


cutlass::half_t alpha0 = cutlass::half_t(1);
cutlass::half_t beta0 = cutlass::half_t(1);
cutlass::gemm::GemmCoord problem_size_0(M, N, K);
typename Gemm0::Arguments arguments_0{
    problem_size_0,
    {reinterpret_cast<cutlass::half_t*>(A0), K}, M * K,
    {reinterpret_cast<cutlass::half_t*>(B0), K}, N * K,
    {reinterpret_cast<cutlass::half_t*>(C0), 0}, 1 * N,
    {reinterpret_cast<cutlass::half_t*>(D0), N}, M * N,
    { alpha0, beta0 },
    Batch};
    Gemm0 gemm_op_0;
    gemm_op_0.initialize(arguments_0, nullptr);
    gemm_op_0(STREAM_CAST(stream));

}
