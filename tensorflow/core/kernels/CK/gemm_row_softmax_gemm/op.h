#pragma once
#include <stdexcept>

#include "ck/ck.hpp"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace functor {
template <typename Device, typename dataTP>
struct GemmRowSoftmaxGemmFunctor {
  static Status Compute(const Device& d, const void* mat_B0, const void* mat_A0,
                        const void* mat_A1, const void* Keymask, void* mat_B1,
                        void* mat_D, int batch, int seq, int head_num,
                        int new_head);
};
}  // namespace functor
}  // namespace tensorflow
