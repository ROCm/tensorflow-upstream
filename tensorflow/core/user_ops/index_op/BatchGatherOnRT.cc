#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <functional>

using namespace tensorflow;

REGISTER_OP("BatchGatherOnRT")
    .Input("params_values: T")
    .Input("params_row_splits: int64")
    .Input("indices_values: int64")
    .Input("indices_row_splits: int64")
    .Output("ret_values: T")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    });



template <typename T>
class BatchGatherOnRT: public OpKernel {
 public:
  explicit BatchGatherOnRT(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const auto params_values = context->input(0).vec<T>();
    const auto params_row_splits = context->input(1).vec<int64>();

    const auto indices_values = context->input(2).vec<int64>();
    const auto indices_row_splits = context->input(3).vec<int64>();

    int num_groups = indices_row_splits.dimension(0) - 1;

    
    Tensor *ret_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {indices_values.dimension(0)}, 
                                                     &ret_values_tensor));
    auto ret_values = ret_values_tensor->vec<T>();
    

    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; i++) {
        for (int j = indices_row_splits(i); j < indices_row_splits(i+1); ++j) {
          int idx = params_row_splits(i) + indices_values(j);
          ret_values(j) = params_values(idx);
        }
      }
    };
    Move(0, num_groups);

  };
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BatchGatherOnRT").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BatchGatherOnRT<T>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
