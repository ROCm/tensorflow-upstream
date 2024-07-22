#pragma once
#include <stdexcept>
#include "ck/ck.hpp"
#include "tensorflow/core/lib/core/errors.h"

#ifndef PARAM_DEFINITION_HPP
#define PARAM_DEFINITION_HPP

struct Param {
    int M;
    int N;
    int K;
    int KBatch;
    int StrideA;
    int StrideB;
    int StrideC;
    const void* A;
    const void* B;
    void* C;
};

#endif // PARAM_DEFINITION_HPP

namespace tensorflow {
namespace functor {
template <typename Device, typename dataTP>
struct Fused_Tile_Gemm_Functor {
    static Status Compute(const Device& d, const Param & param);
};
} // namespace functor
} // namespace tensorflow
