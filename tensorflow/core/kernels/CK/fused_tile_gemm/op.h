#pragma once
#include <stdexcept>

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace functor {
template <typename Device, typename dataTP>
struct FusedTileGemmFunctor {
  static Status Compute(const Device& d, const void* mat_A, const void* mat_B,
                        void* mat_D, int batch, int seq, int head_sz,
                        int head_num);
};
}  // namespace functor
}  // namespace tensorflow
