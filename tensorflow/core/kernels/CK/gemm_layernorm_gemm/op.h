#pragma once
#include <stdexcept>

#include "ck/ck.hpp"
#include "tensorflow/core/lib/core/errors.h"
#ifndef PARAM_DEFINITION_HPP
#define PARAM_DEFINITION_HPP

#endif  // PARAM_DEFINITION_HPP
namespace tensorflow {
namespace functor {
template <typename Device, typename dataTP>
struct GemmLayernormGemmFunctor {
  static Status Compute(const Device& d, const void* mat_A0, const void* mat_B0,
                        const void* mat_C, const void* mat_B1,
                        const void* Gamma, const void* Beta, void* mat_D, int K,
                        int M, int N0, int N1, int head_num, float lrelu_alpha,
                        bool do_layer_norm, bool do_leaky_rel);
};
}  // namespace functor
}  // namespace tensorflow
