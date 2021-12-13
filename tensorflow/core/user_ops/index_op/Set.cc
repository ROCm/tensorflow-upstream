#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace tensorflow;

REGISTER_OP("SetUnion")
    .Input("a_values: T")
    .Input("a_row_splits: int64")
    .Input("b_values: T")
    .Input("b_row_splits: int64")
    .Output("c_values: T")
    .Output("c_row_splits: int64")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &shape));
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

REGISTER_OP("SetIntersection")
    .Input("a_values: T")
    .Input("a_row_splits: int64")
    .Input("b_values: T")
    .Input("b_row_splits: int64")
    .Output("c_values: T")
    .Output("c_row_splits: int64")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &shape));
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

REGISTER_OP("SetDifference")
    .Input("a_values: T")
    .Input("a_row_splits: int64")
    .Input("b_values: T")
    .Input("b_row_splits: int64")
    .Output("c_values: T")
    .Output("c_row_splits: int64")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &shape));
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

REGISTER_OP("BitmapDifference")
    .Input("idx_next: T")
    .Input("idx_flag: int32")
    .Output("idx_next_new: T")
    .Output("idx_flag_new: int32")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &shape));
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

REGISTER_OP("BitmapInit")
    .Input("idx: T")
    .Input("length: int32")
    .Output("bitmap: int32")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      shape_inference::DimensionHandle k_dim;
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &k_dim));
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -1, &s));
      TF_RETURN_IF_ERROR(c->Concatenate(s, c->Vector(k_dim), &s));
      c->set_output(0, s);
      return Status::OK();
    });

template <typename T>
class SetUnion: public OpKernel {
 public:
  explicit SetUnion(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    const auto& a_values = context->input(0).vec<T>();
    const auto& a_row_splits = context->input(1).vec<int64>();
    const auto& b_values = context->input(2).vec<T>();
    const auto& b_row_splits = context->input(3).vec<int64>();

    OP_REQUIRES(context, a_row_splits.dimension(0) == b_row_splits.dimension(0), 
                errors::InvalidArgument("row_splits of two inputs do NOT match: ", 
                                        a_row_splits.dimension(0) ,"!=", b_row_splits.dimension(0)));

    int num_groups = a_row_splits.dimension(0) - 1;

    Tensor *c_row_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, {a_row_splits.dimension(0)}, 
                                                     &c_row_splits_tensor));
    auto c_row_splits = c_row_splits_tensor->vec<int64>();
    c_row_splits(0) = 0;

    std::vector<std::unordered_set<T>> sets(num_groups);

    //[TODO] to be parallized
    std::function<void(int64, int64)> Union = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        sets[i].insert(a_values.data()+a_row_splits(i), a_values.data()+a_row_splits(i+1));
        sets[i].insert(b_values.data()+b_row_splits(i), b_values.data()+b_row_splits(i+1));
      }
    };
    Union(0, num_groups);

    for (int i = 0; i < num_groups; ++i) {
      c_row_splits(i+1) = c_row_splits(i) + sets[i].size();
    }
    Tensor *c_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {c_row_splits(num_groups)}, &c_values_tensor));
    auto c_values = c_values_tensor->vec<T>();

    //[TODO] to be parallized
    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        const auto& set = sets[i];
        std::copy(set.begin(), set.end(), c_values.data()+c_row_splits(i));
      }
    };
    Move(0, num_groups);
  };
};

template <typename T>
class SetIntersection: public OpKernel {
 public:
  explicit SetIntersection(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    const auto& a_values = context->input(0).vec<T>();
    const auto& a_row_splits = context->input(1).vec<int64>();
    const auto& b_values = context->input(2).vec<T>();
    const auto& b_row_splits = context->input(3).vec<int64>();

    OP_REQUIRES(context, a_row_splits.dimension(0) == b_row_splits.dimension(0), 
                errors::InvalidArgument("row_splits of two inputs do NOT match: ", 
                                        a_row_splits.dimension(0) ,"!=", b_row_splits.dimension(0)));

    int num_groups = a_row_splits.dimension(0) - 1;

    Tensor *c_row_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, {a_row_splits.dimension(0)}, 
                                                     &c_row_splits_tensor));
    auto c_row_splits = c_row_splits_tensor->vec<int64>();
    c_row_splits(0) = 0;

    std::vector<std::unordered_set<T>> sets(num_groups);

    //[TODO] to be parallized
    std::function<void(int64, int64)> Intersect = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        std::unordered_set<T> temp(a_values.data()+a_row_splits(i), a_values.data()+a_row_splits(i+1),
                                   a_row_splits(i+1) - a_row_splits(i));
        auto& set = sets[i];
        set.reserve(b_row_splits(i+1) - b_row_splits(i));
        for (int j = b_row_splits(i); j < b_row_splits(i+1); ++j) {
          if (temp.count(b_values(j)))
            set.insert(b_values(j));
        }
      }
    };
    Intersect(0, num_groups);

    for (int i = 0; i < num_groups; ++i) {
      c_row_splits(i+1) = c_row_splits(i) + sets[i].size();
    }
    Tensor *c_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {c_row_splits(num_groups)}, &c_values_tensor));
    auto c_values = c_values_tensor->vec<T>();

    //[TODO] to be parallized
    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        const auto& set = sets[i];
        std::copy(set.begin(), set.end(), c_values.data()+c_row_splits(i));
      }
    };
    Move(0, num_groups);
  };
};

template <typename T>
class SetDifference: public OpKernel {
 public:
  explicit SetDifference(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    const auto& a_values = context->input(0).vec<T>();
    const auto& a_row_splits = context->input(1).vec<int64>();
    const auto& b_values = context->input(2).vec<T>();
    const auto& b_row_splits = context->input(3).vec<int64>();

    OP_REQUIRES(context, a_row_splits.dimension(0) == b_row_splits.dimension(0), 
                errors::InvalidArgument("row_splits of two inputs do NOT match: ", 
                                        a_row_splits.dimension(0) ,"!=", b_row_splits.dimension(0)));

    int num_groups = a_row_splits.dimension(0) - 1;

    Tensor *c_row_splits_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(1, {a_row_splits.dimension(0)}, 
                                                     &c_row_splits_tensor));
    auto c_row_splits = c_row_splits_tensor->vec<int64>();
    c_row_splits(0) = 0;

    std::vector<std::unordered_set<T>> sets(num_groups);

    //[TODO] to be parallized
    std::function<void(int64, int64)> Differ = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        auto& set = sets[i];
        set.insert(a_values.data()+a_row_splits(i), a_values.data()+a_row_splits(i+1));
        for (int j = b_row_splits(i); j < b_row_splits(i+1); ++j) {
          set.erase(b_values(j));
        }
      }
    };
    Differ(0, num_groups);

    for (int i = 0; i < num_groups; ++i) {
      c_row_splits(i+1) = c_row_splits(i) + sets[i].size();
    }
    Tensor *c_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, {c_row_splits(num_groups)}, &c_values_tensor));
    auto c_values = c_values_tensor->vec<T>();

    //[TODO] to be parallized
    std::function<void(int64, int64)> Move = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        const auto& set = sets[i];
        std::copy(set.begin(), set.end(), c_values.data()+c_row_splits(i));
      }
    };
    Move(0, num_groups);
  };
};

template<typename T>
class BitmapDifference: public OpKernel {
 public:
  explicit BitmapDifference(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    const auto& idx_next = context->input(0).vec<T>();
    const auto& idx_flag = context->input(1).vec<int32>();

    int num_idx_next = idx_next.dimension(0);
    int num_idx_flag = idx_flag.dimension(0);


    std::vector<T> idx_next_unvisited;
    idx_next_unvisited.reserve(num_idx_next);
    

    Tensor *idx_flag_new;
    TensorShape idx_flag_output_shape({num_idx_flag});
    OP_REQUIRES_OK(context, context->allocate_output(1, idx_flag_output_shape, &idx_flag_new));
    auto idx_flag_new_values = idx_flag_new->vec<int32>();
    std::copy(idx_flag.data(), idx_flag.data()+num_idx_flag, idx_flag_new_values.data());


    for (int i = 0; i < num_idx_next; ++i){
      T node = idx_next(i);
      int flag_index = node / 32;
      int bit_index = node % 32;
      if (!(idx_flag_new_values(flag_index) & (1 << bit_index))){
        idx_next_unvisited.push_back(node);
        idx_flag_new_values(flag_index) = idx_flag_new_values(flag_index) | (1 << bit_index); 
      }
    }

    Tensor *idx_next_new;
    int num_idx_next_new = idx_next_unvisited.size();
    TensorShape idx_next_output_shape({num_idx_next_new});
    OP_REQUIRES_OK(context, context->allocate_output(0, idx_next_output_shape, &idx_next_new));
    auto idx_next_new_values = idx_next_new->vec<T>();

    std::copy(idx_next_unvisited.begin(), idx_next_unvisited.end(), idx_next_new_values.data());
  };
};


template<typename T>
class BitmapInit: public OpKernel {
 public:
  explicit BitmapInit(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    const auto& idx = context->input(0).vec<T>();
    int length = context->input(1).scalar<int32>()();
    size_t idx_size = idx.size();

    OP_REQUIRES(context, 0 <= length &&  idx_size <= length, 
                errors::InvalidArgument("require: length >= idx.size() and length >=0 but", "length:" ,length, "idx.size():", idx_size));

    Tensor *bitmap;
    TensorShape bitmap_output_shape({length});
    OP_REQUIRES_OK(context, context->allocate_output(0, bitmap_output_shape, &bitmap));
    std::memset(bitmap->flat<int32>().data(), 0, length*sizeof(int32));

    auto bitmap_values = bitmap->vec<int32>();

    for (size_t i = 0; i < idx_size; ++i){
      T node = idx(i);
      int flag_index = node / 32;
      int bit_index = node % 32;
      if (!(bitmap_values(flag_index) & (1 << bit_index))){
        bitmap_values(flag_index) = bitmap_values(flag_index) | (1 << bit_index); 
      }
    }
  };
};

#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(Name("SetUnion").Device(DEVICE_CPU).TypeConstraint<T>("T"), SetUnion<T>); \
  REGISTER_KERNEL_BUILDER(Name("SetIntersection").Device(DEVICE_CPU).TypeConstraint<T>("T"), SetIntersection<T>); \
  REGISTER_KERNEL_BUILDER(Name("SetDifference").Device(DEVICE_CPU).TypeConstraint<T>("T"), SetDifference<T>); \
  REGISTER_KERNEL_BUILDER(Name("BitmapDifference").Device(DEVICE_CPU).TypeConstraint<T>("T"), BitmapDifference<T>); \
  REGISTER_KERNEL_BUILDER(Name("BitmapInit").Device(DEVICE_CPU).TypeConstraint<T>("T"), BitmapInit<T>);

REGISTER_CPU(int32);
REGISTER_CPU(int64);
