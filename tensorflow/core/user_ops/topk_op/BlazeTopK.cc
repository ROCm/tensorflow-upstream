#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/work_sharder.h"
#include <algorithm>
#include <cmath>

using namespace tensorflow;

REGISTER_OP("BlazeTopK")
    .Input("input: T")              //[..., input_len]
    .Input("k: Tindices")           //scaler
    .Output("value: T")             //[..., k]
    .Output("index: Tindices")      //[..., k]
    .Attr("T: {half, float, double}")
    .Attr("Tindices: {int32}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(0), -1, c->UnknownDim(), &output));
      c->set_output(0, output);
      return Status::OK();
    });





//Impl of grouped topk algorithm
//The third input splits is a vector containing length of each group.
template <typename T>
class BlazeTopK : public OpKernel {
 public:
  explicit BlazeTopK(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //output[dst_idx[i]:dst_idx[i+1]) = topk of input[src_idx[i]:src_idx[i+1])
    const Tensor & input_tensor = context->input(0);
    const auto& input = input_tensor.flat_inner_dims<T>();

    int k = context->input(1).scalar<int>()();

    int batch_size = input.dimension(0);
    int input_len = input.dimension(1);

    OP_REQUIRES(context, 0 <= k && k <= input_len, 
                errors::InvalidArgument("require: 0 <= k <= input_len, but", k ," > ", input_len));

    //Allocate Output
    TensorShape output_shape = input_tensor.shape();
    output_shape.set_dim(output_shape.dims()-1, k);
    Tensor *value_output, *index_output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &value_output));
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &index_output));
    auto value = value_output->flat_inner_dims<T>();
    auto index = index_output->flat_inner_dims<int>();

    if (k == 0)
      return;

    int sampling_num = std::min(std::max(1000, k), input_len);
    float ratio = (float)k / input_len;
    int idx0 = std::min((int)std::ceil(ratio*sampling_num)*2, sampling_num-1);

    std::function<void(int64, int64)> shard = [&](int64 begin, int64 end) {
      //outside for loop for reuse
      std::vector<int> candidate_idx;
      candidate_idx.reserve(4*k);

      for (int batch = begin; batch < end; ++batch) {
        
        const T* input_v = input.data() + batch * input_len;
        std::vector<T> sample(input_v, input_v+sampling_num);
        std::sort(sample.begin(), sample.end(), 
                  [](T a, T b) { return a > b; });

        int idx = idx0 - 1;
        do {
          idx += 1;
          candidate_idx.clear();
          T threshold = sample[idx];
          for (int i = 0; i < input_len; ++i) {
            if (input_v[i] > threshold) {
              candidate_idx.push_back(i);
            }
          }
        } while (candidate_idx.size() < k);
        
        std::partial_sort(candidate_idx.begin(), candidate_idx.begin()+k, candidate_idx.end(),
                          [&input_v](int a, int b) { return input_v[a] > input_v[b]; });

        //copy candidate to ouput buffer
        T* value_v = value.data() + batch * k;
        int* index_v = index.data() + batch * k;
        for (int i = 0; i < k; ++i) {
          index_v[i] = candidate_idx[i];
          value_v[i] = input_v[index_v[i]];
        }
      }
    };

    const DeviceBase::CpuWorkerThreads* worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int64 kCostPerBatch = 3*sampling_num*std::log(sampling_num) + 6*k*std::log(k) + 3*input_len;
    Shard(worker_threads->num_threads, worker_threads->workers, batch_size, kCostPerBatch, shard);
  };
};


// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("BlazeTopK").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BlazeTopK<T>);
REGISTER_CPU(double);
REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);
