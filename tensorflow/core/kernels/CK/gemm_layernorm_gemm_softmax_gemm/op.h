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
struct GemmLayernormGemmSoftmaxGemmFunctor {
  static Status Compute(const Device& d, const void* mat_A0, const void* mat_B0,
                        const void* mat_C, const void* mat_B1,
                        const void* mat_B2, const void* Gamma, const void* Beta,
                        const void* Keymask, void* mat_D, int K, int M, int N0,
                        int N1, int long_seq, int N2, int B_kv, int head_num,
                        float lrelu_alpha, bool do_layer_norm,
                        bool do_leaky_relu, bool do_query_mas);
};
}  // namespace functor
}  // namespace tensorflow
