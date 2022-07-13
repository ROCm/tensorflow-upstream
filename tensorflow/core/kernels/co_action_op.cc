//
// Created by qiaoxj on 2020/9/7.
//

#include "tensorflow/core/kernels/co_action_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

template <typename T>
Status PowCPU(const CPUDevice& d, size_t m, size_t n, const T* in, T* out,
              int64 pow_num) {
  typename tensorflow::TTypes<const T>::Matrix in_matrix(in, m, n);
  typename tensorflow::TTypes<T>::Matrix out_matrix(out, m, n);
  out_matrix.device(d) = in_matrix.pow((T)pow_num);
  return Status::OK();
}

template <typename T>
Status CoActionCPU(const CPUDevice& d, size_t m, size_t n, size_t k, const T* a,
                   const T* b, T* c) {
  typename tensorflow::TTypes<const T>::Matrix a_matrix(a, m, k);
  typename tensorflow::TTypes<const T>::Matrix b_matrix(b, k, n);
  typename tensorflow::TTypes<T>::Matrix c_matrix(c, 1, n);
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
  dim_pair[0].first = 1;
  dim_pair[0].second = 0;
  Eigen::array<int, 1> reduce_dims({0});
  c_matrix.device(d) =
      a_matrix.contract(b_matrix, dim_pair).tanh().sum(reduce_dims);
  return Status::OK();
}

template <typename Scalar>
struct LaunchCoAction<CPUDevice, Scalar> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                    const Tensor& in_a, const Tensor& in_b, Tensor* out,
                    int64 batch_a, int64 batch_b, int64 paralle_num,
                    int64 pow_num) {
    auto a_ptr = in_a.template flat<Scalar>().data();
    auto b_ptr = in_b.template flat<Scalar>().data();
    auto c_ptr = out->template flat<Scalar>().data();
    Tensor tmp_pow;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Scalar>::value,
        TensorShape({batch_a, paralle_num, pow_num * m, k}), &tmp_pow));
    auto tmp_pow_ptr = tmp_pow.template flat<Scalar>().data();
    // power and concat
    for (int64 batch = 0; batch < batch_a; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(PowCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, k,
              a_ptr + (batch * paralle_num + p) * m * k,
              tmp_pow_ptr + (batch * paralle_num + p) * pow_num * m * k +
                  pow * m * k,
              pow + 1));
        }
      }
    }

    for (int64 batch = 0; batch < batch_b; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(CoActionCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, n, k,
              tmp_pow_ptr + (p * pow_num + pow) * m * k,
              b_ptr + (batch * paralle_num + p) * k * n,
              c_ptr + ((batch * paralle_num + p) * pow_num + pow) * 1 * n));
        }
      }
    }
    return Status::OK();
  }
};

template <typename Scalar, typename TIndex>
struct LaunchCoActionIndicator<CPUDevice, Scalar, TIndex> {
  Status operator()(OpKernelContext* context, int64 m, int64 n, int64 k,
                    const Tensor& in_a, const Tensor& in_b,
                    const Tensor& indicator, Tensor* out, int64 batch_a,
                    int64 batch_b, int64 paralle_num, int64 pow_num) {
    auto a_ptr = in_a.template flat<Scalar>().data();
    auto b_ptr = in_b.template flat<Scalar>().data();
    auto c_ptr = out->template flat<Scalar>().data();
    auto ind_ptr = indicator.template flat<TIndex>().data();
    Tensor tmp_pow;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<Scalar>::value,
        TensorShape({batch_a, paralle_num, pow_num * m, k}), &tmp_pow));
    auto tmp_pow_ptr = tmp_pow.template flat<Scalar>().data();

    // power and concat
    for (int64 batch = 0; batch < batch_a; batch++) {
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(PowCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, k,
              a_ptr + (batch * paralle_num + p) * m * k,
              tmp_pow_ptr + (batch * paralle_num + p) * pow_num * m * k +
                  pow * m * k,
              pow + 1));
        }
      }
    }
    for (int64 batch = 0; batch < batch_b; batch++) {
      auto ind = (int64)ind_ptr[batch];
      if (ind < 0 || ind >= batch_a) {
        // printf("Indicator ERROR for indicator_matmul, indicator: %d.\n",
        // ind);
        ind = 0;
      }
      for (int64 p = 0; p < paralle_num; p++) {
        for (int64 pow = 0; pow < pow_num; pow++) {
          TF_RETURN_IF_ERROR(CoActionCPU<Scalar>(
              context->eigen_device<CPUDevice>(), m, n, k,
              tmp_pow_ptr + ((ind * paralle_num + p) * pow_num + pow) * m * k,
              b_ptr + (batch * paralle_num + p) * k * n,
              c_ptr + ((batch * paralle_num + p) * pow_num + pow) * 1 * n));
        }
      }
    }
    return Status::OK();
  }
};

template <typename Device, typename Scalar>
class CoActionOp : public OpKernel {
 public:
  explicit CoActionOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pow_num", &pow_num));
  }

  ~CoActionOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& a = ctx->input(0);
    auto& b = ctx->input(1);

    OP_REQUIRES(ctx, a.dims() == 4,
                errors::InvalidArgument("In[0] ndims must be 4: ", a.dims()));
    OP_REQUIRES(ctx, b.dims() == 4,
                errors::InvalidArgument("In[1] ndims must be 4: ", b.dims()));
    // currently only support m=150/50, k=5, n=4, pow=2
    OP_REQUIRES(ctx, pow_num == 2,
                errors::InvalidArgument("pow_num must == 2: ", pow_num));
    OP_REQUIRES(
        ctx, a.dim_size(2) == 50 || a.dim_size(2) == 150,
        errors::InvalidArgument("m must be 50 or 150: ", a.dim_size(2)));
    OP_REQUIRES(ctx, b.dim_size(2) == 5,
                errors::InvalidArgument("k must be 5: ", b.dim_size(2)));
    OP_REQUIRES(ctx, b.dim_size(3) == 4,
                errors::InvalidArgument("n must be 4: ", b.dim_size(3)));

    int64 d0 = a.dim_size(2);
    int64 d1 = a.dim_size(3);
    int64 d2 = b.dim_size(2);
    int64 d3 = b.dim_size(3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("a mismatch b shape: ", d1, " vs. ", d2,
                                        ": ", a.shape().DebugString(), " ",
                                        b.shape().DebugString()));
    int64 batch_a = a.dim_size(0);
    int64 batch_b = b.dim_size(0);
    OP_REQUIRES(ctx, batch_a == 1,
                errors::InvalidArgument("batch_a must be 1: a_shape = ",
                                        a.shape().DebugString()));
    int64 parallel_a = a.dim_size(1);
    int64 parallel_b = b.dim_size(1);
    OP_REQUIRES(ctx, parallel_a == parallel_b,
                errors::InvalidArgument(
                    "parallel_a mismatch parallel_b : ", parallel_a, " vs. ",
                    parallel_b, ": ", a.shape().DebugString(), " ",
                    b.shape().DebugString()));
    TensorShape out_shape({batch_b, parallel_a, pow_num, d3});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (a.NumElements() == 0 || b.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }

    //[PROF-STATS]
    int64 delta = 2 * d1 * out_shape.num_elements() * pow_num;
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "FLOPs = " << delta
                << ", " << type_string()
                << ", " << name()
                << ", " << a.shape().DebugString()
                << ", " << b.shape().DebugString();
    }

    OP_REQUIRES_OK(ctx, LaunchCoAction<Device, Scalar>()(ctx, d0, d3, d1, a, b,
                                                         out, batch_a, batch_b,
                                                         parallel_a, pow_num));
  }

 private:
  int64 pow_num;
};

template <typename Device, typename Scalar, typename TIndex>
class CoActionIndicatorOp : public OpKernel {
 public:
  explicit CoActionIndicatorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pow_num", &pow_num));
  }

  ~CoActionIndicatorOp() = default;

  void Compute(OpKernelContext* ctx) override {
    auto& a = ctx->input(0);
    auto& b = ctx->input(1);
    auto& ind = ctx->input(2);

    OP_REQUIRES(ctx, a.dims() == 4,
                errors::InvalidArgument("In[0] ndims must be 4: ", a.dims()));
    OP_REQUIRES(ctx, b.dims() == 4,
                errors::InvalidArgument("In[1] ndims must be 4: ", b.dims()));
    OP_REQUIRES(ctx, ind.dims() == 1,
                errors::InvalidArgument("In[2] ndims must be 1: ", ind.dims()));
    // currently only support m=150/50, k=5, n=4, pow=2
    OP_REQUIRES(
        ctx, a.dim_size(2) == 50 || a.dim_size(2) == 150,
        errors::InvalidArgument("m must be 50 or 150: ", a.dim_size(2)));
    OP_REQUIRES(ctx, pow_num == 2,
                errors::InvalidArgument("pow_num must == 2: ", pow_num));
    OP_REQUIRES(ctx, b.dim_size(2) == 5,
                errors::InvalidArgument("k must be 5: ", b.dim_size(2)));
    OP_REQUIRES(ctx, b.dim_size(3) == 4,
                errors::InvalidArgument("n must be 4: ", b.dim_size(3)));

    int64 d0 = a.dim_size(2);
    int64 d1 = a.dim_size(3);
    int64 d2 = b.dim_size(2);
    int64 d3 = b.dim_size(3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("a mismatch b shape: ", d1, " vs. ", d2,
                                        ": ", a.shape().DebugString(), " ",
                                        b.shape().DebugString()));
    int64 parallel_a = a.dim_size(1);
    int64 parallel_b = b.dim_size(1);
    OP_REQUIRES(ctx, parallel_a == parallel_b,
                errors::InvalidArgument(
                    "parallel_a mismatch parallel_b : ", parallel_a, " vs. ",
                    parallel_b, ": ", a.shape().DebugString(), " ",
                    b.shape().DebugString()));
    int64 batch_a = a.dim_size(0);
    int64 batch_b = b.dim_size(0);
    int64 ind_length = ind.dim_size(0);
    OP_REQUIRES(
        ctx, batch_b == ind_length,
        errors::InvalidArgument(
            "b_batch mismatch indicator length: ", batch_b, " vs. ", ind_length,
            ": ", b.shape().DebugString(), " ", ind.shape().DebugString()));
    TensorShape out_shape({batch_b, parallel_a, pow_num, d3});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (a.NumElements() == 0 || b.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }

    //[PROF-STATS]
    int64 delta = 2 * d1 * out_shape.num_elements() * pow_num;
    if (VLOG_IS_ON(1)) {
      LOG(INFO) << "FLOPs = " << delta
                << ", " << type_string()
                << ", " << name()
                << ", " << a.shape().DebugString()
                << ", " << b.shape().DebugString();
    }

    OP_REQUIRES_OK(ctx, LaunchCoActionIndicator<Device, Scalar, TIndex>()(
                            ctx, d0, d3, d1, a, b, ind, out, batch_a, batch_b,
                            parallel_a, pow_num));
  }

 private:
  int64 pow_num;
};

#define REGISTER_COACTION_CPU(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("CoAction").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),    \
      CoActionOp<CPUDevice, TYPE>);                                     \
  REGISTER_KERNEL_BUILDER(Name("CoActionIndicator")                     \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<TYPE>("T")                \
                              .TypeConstraint<int32>("Tindices"),       \
                          CoActionIndicatorOp<CPUDevice, TYPE, int32>); \
  REGISTER_KERNEL_BUILDER(Name("CoActionIndicator")                     \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<TYPE>("T")                \
                              .TypeConstraint<int64>("Tindices"),       \
                          CoActionIndicatorOp<CPUDevice, TYPE, int64>);

REGISTER_COACTION_CPU(float);
REGISTER_COACTION_CPU(Eigen::half);

#if GOOGLE_CUDA
#define REGISTER_COACTION_GPU(TYPE)                                       \
  extern template struct LaunchCoAction<GPUDevice, TYPE>;                 \
  extern template struct LaunchCoActionIndicator<GPUDevice, TYPE, int32>; \
  extern template struct LaunchCoActionIndicator<GPUDevice, TYPE, int64>; \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("CoAction").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),      \
      CoActionOp<GPUDevice, TYPE>);                                       \
  REGISTER_KERNEL_BUILDER(Name("CoActionIndicator")                       \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<TYPE>("T")                  \
                              .TypeConstraint<int32>("Tindices"),         \
                          CoActionIndicatorOp<GPUDevice, TYPE, int32>);   \
  REGISTER_KERNEL_BUILDER(Name("CoActionIndicator")                       \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<TYPE>("T")                  \
                              .TypeConstraint<int64>("Tindices"),         \
                          CoActionIndicatorOp<GPUDevice, TYPE, int64>);

REGISTER_COACTION_GPU(float);
REGISTER_COACTION_GPU(Eigen::half);
#endif

}  // namespace tensorflow
