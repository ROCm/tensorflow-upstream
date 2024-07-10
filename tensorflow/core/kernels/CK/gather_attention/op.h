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
struct GatherAttentionFunctor {
  static Status Compute(const Device& d, const void* mat_A, const void* mat_B0,
                        const void* keymask, const int* indices,
                        const void* mat_B1, void* mat_D, int head_sz, int seq,
                        int B, int index, int head_num);
};
}  // namespace functor
}  // namespace tensorflow
