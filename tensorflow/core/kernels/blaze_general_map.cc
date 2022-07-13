#include <map>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/numbers.h"
namespace tensorflow {
template<typename IN, typename OUT>
class BlazeGeneralMap : public OpKernel {
 public:
  explicit BlazeGeneralMap(OpKernelConstruction* ctx) : OpKernel(ctx) 
  {
    std::vector<string> keys;
    std::vector<string> values;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keys", &keys));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("values", &values));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_val", &default_val_));
    OP_REQUIRES_OK(ctx, InitMap(keys, values));
  }
  void Compute(OpKernelContext* ctx) {
    auto& input = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(input.shape()),
                errors::Internal("input not saclar"));
    auto& key = input.scalar<IN>()();
    auto iter = map_.find(key);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    if (iter == map_.end()) {
      out->scalar<OUT>()() = default_val_;
    } else {
      out->scalar<OUT>()() = map_[key];
    }
  }
 private:
  Status InitMap(const std::vector<string>& keys, const std::vector<string>& values) {
    if (keys.size() != values.size()) {
      return errors::Internal("key size ", keys.size(), " not euqal to value size ",
                              values.size());
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      IN in;
      StringPiece sp(keys[i]);
      if (!strings::SafeStringToNumeric<IN>(sp, &in)) {
        return errors::Internal("Parse ", keys[i], " failed" );
      }
      StringPiece outsp(values[i]);
      OUT out;
      if (!strings::SafeStringToNumeric<OUT>(outsp, &out)) {
        return errors::Internal("Parse ", values[i], " failed");
      }
      map_[in] = out;
    }
    return Status::OK();
  }
 private:
  std::map<IN, OUT> map_;
  OUT default_val_;
};
#define REGISTER_KERNEL_GENERAL_MAP(in_type, out_type)       \
  REGISTER_KERNEL_BUILDER(Name("BlazeGeneralMap")               \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<in_type>("Tin")       \
                              .TypeConstraint<out_type>("Tout"), \
                          BlazeGeneralMap<in_type, out_type>);
REGISTER_KERNEL_GENERAL_MAP(int64, int32);
}
