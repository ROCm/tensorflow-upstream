#pragma once
#include <stdexcept>
#include "ck/ck.hpp"
#include "tensorflow/core/lib/core/errors.h"
#ifndef PARAM_DEFINITION_HPP
#define PARAM_DEFINITION_HPP

struct Param {
    // Define the members of Param here
};

#endif // PARAM_DEFINITION_HPP
namespace tensorflow {
namespace functor {
template <typename Device, typename dataTP>
struct Fused_Gemm_Bias_Add_Functor {
    static Status Compute(const Device& d, const Param & param);
};
} // namespace functor
} // namespace tensorflow
