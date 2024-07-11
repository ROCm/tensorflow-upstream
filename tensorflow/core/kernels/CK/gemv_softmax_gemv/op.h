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
struct GemvSoftmaxGemvFunctor {
  static Status Compute(const Device& d, const void* mat_A, const void* mat_B0,
                        const void* keymask, const void* mat_B1, void* mat_D,
                        int head_sz, int seq, int b);
};
}  // namespace functor
}  // namespace tensorflow
