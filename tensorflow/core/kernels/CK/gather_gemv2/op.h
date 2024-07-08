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
struct GatherGemv2Functor {
  static Status Compute(const Device& d, const void* mat_A, const void* mat_B,
                        const int* indices, void* mat_D, int head_sz, int seq,
                        int B, int index, int head_num);
};
}  // namespace functor
}  // namespace tensorflow
