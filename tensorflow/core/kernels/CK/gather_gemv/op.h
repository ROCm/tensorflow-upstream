#pragma once
#include <stdexcept>

#include "ck/ck.hpp"
#include "tensorflow/core/lib/core/errors.h"
 
namespace tensorflow {
namespace functor {
template <typename Device, typename dataTP>
struct GatherGemvFunctor {
  static Status Compute(const Device& d, const void* mat_A, const void* mat_B,
                        const int* indices, void* mat_D, int head_sz, int seq,
                        int B, int index, int head_num);
};
}  // namespace functor
}  // namespace tensorflow
