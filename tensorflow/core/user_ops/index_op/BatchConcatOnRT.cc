#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <functional>

using namespace tensorflow;


REGISTER_OP("BatchConcatOnRT")
    .Input("left_values: T")
    .Input("left_row_splits: int64")
    .Input("right_values: T")
    .Input("right_row_splits: int64")
    .Output("ret_values: T")
    .Output("ret_row_splits: int64")
    .Attr("T: {half, float, double, int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });



template <typename T>
class BatchConcatOnRT: public OpKernel {
 public:
  explicit BatchConcatOnRT(OpKernelConstruction* context) : OpKernel(context) {
  }
  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const auto left_values = context->input(0).vec<T>();
    const auto left_row_splits = context->input(1).vec<int64>();

    const auto right_values = context->input(2).vec<T>();
    const auto right_row_splits = context->input(3).vec<int64>();

    int num_groups = left_row_splits.dimension(0) - 1;


    Tensor *ret_values_tensor;
    Tensor *ret_row_splits_tensor;

    OP_REQUIRES_OK(context, context->allocate_output(0, {left_values.dimension(0) + right_values.dimension(0)},
                                                     &ret_values_tensor));

    OP_REQUIRES_OK(context, context->allocate_output(1, {left_row_splits.dimension(0)},
                                                     &ret_row_splits_tensor));

    auto ret_values = ret_values_tensor->vec<T>();
    auto ret_row_splits = ret_row_splits_tensor->vec<int64>();

    ret_row_splits(0) = 0;

    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; i++) {
        int left_begin = left_row_splits(i);
        int left_end = left_row_splits(i+1);
        int right_begin = right_row_splits(i);
        int right_end = right_row_splits(i+1);
        std::copy(left_values.data()+left_begin, left_values.data()+left_end, ret_values.data()+right_begin+left_begin);
        std::copy(right_values.data()+right_begin, right_values.data()+right_end, ret_values.data()+right_begin+left_end);
        ret_row_splits(i+1) = left_end + right_end;
      }
    };
    Move(0, num_groups);
  };
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BatchConcatOnRT").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BatchConcatOnRT<T>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(Eigen::half);
