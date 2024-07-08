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
struct FusedGemmBiasAddFunctor {
  static Status Compute(const Device& d, int M, int N, int K, int Batch,
                        const void* a0, const void* b0, const void* d0,
                        void* e);
};
}  // namespace functor
}  // namespace tensorflow
