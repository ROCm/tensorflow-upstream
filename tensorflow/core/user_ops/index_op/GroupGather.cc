#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace tensorflow;

REGISTER_OP("GroupGather")
    .Input("params_values: T")
    .Input("params_row_splits: int64")
    .Input("indices_values: int64")
    .Input("indices_row_splits: int64")
    .Output("ret_values: T")
    .Output("ret_row_splits: int64")
    .Attr("T: {int32, int64}")
    .Attr("unique: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(3));
      return Status::OK();
    });

template <typename T>
class GroupGather: public OpKernel {
 private:
  bool unique_ = false;
 public:
  explicit GroupGather(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("unique", &unique_));
  }

  void Compute(OpKernelContext* context) override {
    //a list of nodes whose children will be returned
    const auto params_values = context->input(0).vec<T>();
    const auto params_row_splits = context->input(1).vec<int64>();

    const auto indices_values = context->input(2).vec<int64>();
    const auto indices_row_splits = context->input(3).vec<int64>();

    int num_groups = indices_row_splits.dimension(0) - 1;

    
    Tensor *ret_row_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, {indices_row_splits.dimension(0)}, 
                                                     &ret_row_splits_tensor));
    auto ret_row_splits = ret_row_splits_tensor->vec<int64>();
    ret_row_splits(0) = 0;

    //==================================
    // UNIQUE OP ON EACH GROUP
    //==================================
    if (unique_) {
      std::vector<std::unordered_set<T>> sets(num_groups);

      //[TODO] to be parallized
      std::function<void(int64, int64)> Gather = [&](int64 begin, int64 end) {
        for (int i = begin; i < end; ++i) {
          for (int j = indices_row_splits(i); j < indices_row_splits(i+1); ++j) {
            int group_idx = indices_values(j);
            sets[i].insert(params_values.data()+params_row_splits(group_idx),
                             params_values.data()+params_row_splits(group_idx+1));
          }
        }
      };
      Gather(0, num_groups);

      int sum = 0;
      for (int i = 0; i < num_groups; ++i) {
        sum += sets[i].size();
        ret_row_splits(i+1) = sum;
      }
      Tensor *ret_values_tensor;
      OP_REQUIRES_OK(context, context->allocate_output(0, {sum}, &ret_values_tensor));
      auto ret_values = ret_values_tensor->vec<T>();

      //[TODO] to be parallized
      std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
        for (int i = begin; i < end; ++i) {
          const auto& set = sets[i];
          int value_idx = ret_row_splits(i);
          for (const T& item: set) {
            ret_values(value_idx) = item;
            value_idx++;
          }
          OP_REQUIRES(context, value_idx == ret_row_splits(i+1), 
                    errors::InvalidArgument("Number of collected values does NOT match number in row_splits: ", 
                                            value_idx ,"!=", ret_row_splits(i+1), " @group ", i));
        }
      };
      Move(0, num_groups);

    }

    //==================================
    // NO UNIQUE OP ON EACH GROUP
    //==================================
    else {
      int sum = 0;
      for (int i = 0; i < num_groups; ++i) {
        for (int j = indices_row_splits(i); j < indices_row_splits(i+1); ++j) {
          int idx = indices_values(j);
          int len = params_row_splits(idx+1) - params_row_splits(idx);
          sum += len;
        }
        ret_row_splits(i+1) = sum;
      }

      Tensor *ret_values_tensor;
      OP_REQUIRES_OK(context, context->allocate_output(0, {sum}, &ret_values_tensor));
      auto ret_values = ret_values_tensor->vec<T>();

      //[TODO] to be parallized
      std::function<void(int64, int64)> Fill = [&](int64 begin, int64 end) {
        for (int i = begin; i < end; ++i) {
          int value_idx = ret_row_splits(i);
          for (int j = indices_row_splits(i); j < indices_row_splits(i+1); ++j) {
            int group_idx = indices_values(j);
            for (int k = params_row_splits(group_idx); k < params_row_splits(group_idx+1); ++k) {
              ret_values(value_idx) = params_values(k);
              value_idx++;
            }
          }
          //assert values_idx == ret_row_splits(i+1)
          OP_REQUIRES(context, value_idx == ret_row_splits(i+1), 
                    errors::InvalidArgument("Number of collected values does NOT match number in row_splits: ", 
                                            value_idx ,"!=", ret_row_splits(i+1), " @group ", i));
        }
      };
      Fill(0, num_groups);

    } // end of else: NO UNIQUE OP ON EACH GROUP


  };
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("GroupGather").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      GroupGather<T>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
