#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace tensorflow;

REGISTER_OP("SplitsGather")
    .Input("splits: T")
    .Input("indices_values: int64")
    .Input("indices_row_splits: int64")
    .Output("ret_values: T")
    .Output("ret_row_splits: int64")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(2));
      return Status::OK();
    });

template <typename T>
class SplitsGather: public OpKernel {
 public:
  explicit SplitsGather(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const auto splits = context->input(0).vec<T>();

    const auto indices_values = context->input(1).vec<int64>();
    const auto indices_row_splits = context->input(2).vec<int64>();

    int num_groups = indices_row_splits.dimension(0) - 1;

    Tensor *ret_row_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, {indices_row_splits.dimension(0)}, 
                                                     &ret_row_splits_tensor));
    auto ret_row_splits = ret_row_splits_tensor->vec<int64>();

    int sum = 0;
    ret_row_splits(0) = 0;
    for (int i = 0; i < num_groups; ++i) {
      for (int j = indices_row_splits(i); j < indices_row_splits(i+1); ++j) {
        int idx = indices_values(j);
        int len = splits(idx+1) - splits(idx);
        sum += len;
      }
      ret_row_splits(i+1) = sum;
    }

    Tensor *ret_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {sum}, &ret_values_tensor));
    auto ret_values = ret_values_tensor->vec<T>();

    // can be parallelize on i
    for (int i = 0; i < num_groups; ++i) {
      int value_idx = ret_row_splits(i);
      for (int j = indices_row_splits(i); j < indices_row_splits(i+1); ++j) {
        int group_idx = indices_values(j);
        for (int k = splits(group_idx); k < splits(group_idx+1); ++k) {
          ret_values(value_idx) = k;
          value_idx++;
        }
      }
      //assert values_idx == ret_row_splits(i+1)
      OP_REQUIRES(context, value_idx == ret_row_splits(i+1),
                errors::InvalidArgument("Number of collected values does NOT match number in row_splits: ", 
                                        value_idx ,"!=", ret_row_splits(i+1), " @group ", i));
    }
  };
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("SplitsGather").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SplitsGather<T>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
