#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"
#include <algorithm>
#include <vector>
#include <iostream>

using namespace tensorflow;


REGISTER_OP("BatchTopKOnRT")
    .Input("values_in: T")
    .Input("row_splits_in: int64")
    .Input("k: int64")
    .Output("values_out: T")
    .Output("idx_out: int64")
    .Output("row_splits_out: int64")
    .Attr("T: {double, float, half}")
    .Attr("ascending: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle values_in, row_splits_in, k;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &values_in));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &row_splits_in));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &k));
      if (c->Rank(k) == 1 && 
          c->Value(c->Dim(k, 0)) != c->Value(c->Dim(row_splits_in, 0)) - 1) {
        return errors::InvalidArgument("length of k != number of groups: ",
                                       c->Value(c->Dim(k, 0)), " != ", c->Value(c->Dim(row_splits_in, 0)), " - 1");
      }
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      c->set_output(2, c->input(1));
      return Status::OK();
    });

template <typename T>
class BatchTopKOnRT : public OpKernel {
 private:
  bool ascending_ = false;

 public:
  explicit BatchTopKOnRT(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("ascending", &ascending_));
  };

  void Compute(OpKernelContext* context) override {
    
    const Tensor& values_tensor_in = context->input(0);
    const auto values_in = values_tensor_in.vec<T>();

    const Tensor& row_splits_tensor_in = context->input(1);
    const auto row_splits_in = row_splits_tensor_in.vec<int64>();

    const Tensor& k_tensor_in = context->input(2);

    int values_len = values_in.dimension(0);
    int num_groups = row_splits_in.dimension(0)-1;

    OP_REQUIRES(context, row_splits_in(num_groups) == values_len, 
                errors::InvalidArgument("Bound of splits does NOT match size of values: ", 
                                        row_splits_in(num_groups) ,"!=", values_len));

    std::vector<int64> k;
    if (k_tensor_in.dims() == 0) {
      const auto k_in_s = k_tensor_in.scalar<int64>();
      k = std::vector<int64>(num_groups, k_in_s());
    } else {
      const auto k_in_v = k_tensor_in.vec<int64>();
      OP_REQUIRES(context, k_in_v.dimension(0) == num_groups, 
                errors::InvalidArgument("Size of k vector does NOT match number of groups: ", 
                                        k_in_v.dimension(0) ,"!=", num_groups));
      k = std::vector<int64>(k_in_v.data(), k_in_v.data()+num_groups);
    }

    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "BatchTopKOnRT: " 
                << ", values_len=" << values_len
                << ", num_groups=" << num_groups;
    }

    // prepare output
    // Allocate Output
    Tensor *row_splits_tensor_out;
    OP_REQUIRES_OK(context, context->allocate_output(2, row_splits_tensor_in.shape(), &row_splits_tensor_out));
    auto row_splits_out = row_splits_tensor_out->vec<int64>();
    row_splits_out(0) = 0;
    int64 sum = 0; 
    for (size_t i = 0; i < num_groups; i++) {
      int64 len = row_splits_in(i+1) - row_splits_in(i);
      sum += std::min(len, k[i]);
      row_splits_out(i+1) = sum;
    }
 
    Tensor *values_tensor_out, *idxes_tensor_out;
    OP_REQUIRES_OK(context, context->allocate_output(0, {sum}, &values_tensor_out));
    OP_REQUIRES_OK(context, context->allocate_output(1, {sum}, &idxes_tensor_out));
    auto values_out = values_tensor_out->vec<T>();
    auto idxes_out = idxes_tensor_out->vec<int64>();

    // partial_sort_copy
    std::vector<size_t> idx(values_len);
    std::iota(idx.begin(), idx.end(), 0);
    std::function<void(int64, int64)> top_k = [&](int64 begin, int64 end) {
      for (int i = begin; i < end; ++i) {
        int head_in = row_splits_in(i);
        int tail_in = row_splits_in(i+1);
        int head_out = row_splits_out(i);
        int tail_out = row_splits_out(i+1);
        if (ascending_)
          std::partial_sort_copy(idx.begin()+head_in, idx.begin()+tail_in,
                                 idxes_out.data()+head_out, idxes_out.data()+tail_out,
                                 [&values_in](int a, int b) { return values_in(a) < values_in(b); });
        else
          std::partial_sort_copy(idx.begin()+head_in, idx.begin()+tail_in,
                                 idxes_out.data()+head_out, idxes_out.data()+tail_out,
                                 [&values_in](int a, int b) { return values_in(a) > values_in(b); });
        for (int j = head_out; j < tail_out; ++j) {
          values_out(j) = values_in(idxes_out(j));
          idxes_out(j) -= head_in;
        }
      }
    };
    
    //top_k(0, num_groups);
    const DeviceBase::CpuWorkerThreads* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;
    int64 kCostPerGroup = 4 * values_len * sum / (num_groups*num_groups);
    Shard(num_threads, worker_threads->workers, num_groups, kCostPerGroup, top_k);
  };
};


// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BatchTopKOnRT").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BatchTopKOnRT<T>);
REGISTER_CPU(double);
REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);
