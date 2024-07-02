#pragma once
#include "ck/ck.hpp"
#include "tensorflow/core/lib/core/errors.h"
namespace tensorflow {
namespace functor {
template <typename Device, typename dataTP>
struct Fused_Gemm_Bias_Add_Functor {
    static Status Compute(const Device& d, int sm, const Param & param);
};
} // namespace functor
} // namespace tensorflow
