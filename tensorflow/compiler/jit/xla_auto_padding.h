#ifndef TENSORFLOW_COMPILER_JIT_XLA_AUTO_PADDING_H_
#define TENSORFLOW_COMPILER_JIT_XLA_AUTO_PADDING_H_

#include <atomic>
#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/compiler/jit/xla_argument_dumper.h"

namespace tensorflow {
class XlaCompilationCache;

class XlaAutoPadding {
public:
  XlaAutoPadding(XlaCompilationCache* cache, std::string name);

  Status FillAndFindCacheShape(
    const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    std::vector<XlaCompiler::Argument>* args,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info);

  Status Compile(
    const XlaCompiler::Options& options, const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args,
    const XlaCompiler::CompileOptions& compile_options,
    const std::function<Status(XlaCompiler* compiler,
            XlaCompiler::CompilationResult*,
            const XlaCompiler::CompileOptions& compile_options,
            const NameAttrList& function,
            absl::Span<const XlaCompiler::Argument> args,
            std::shared_ptr<InputsShapeInfo> inputs_shape_info)>& compile_fn,
    absl::optional<int64> compile_threshold,
    const XlaCompiler::CompilationResult** out_compilation_result,
    xla::LocalExecutable** out_executable,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info);

  Status CompileCallback(
    const XlaCompiler::Options& options,
    absl::Span<const XlaCompiler::Argument> args,
    XlaCompiler::CompilationResult* out_compilation_result,
    xla::LocalExecutable* out_executable,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info);

private:
  int GetCachedSize() {
    mutex_lock lock(cached_input_shapes_mu_);
    return cached_inputs_shapes_.size();
  }
  inline Status AdjustBeyondMax(
     std::shared_ptr<InputsShapeInfo> inputs,
     const std::vector<int>& beyond_max_index);
  void ConvertXLAShapeToTensors(
      const absl::Span<const XlaCompiler::Argument>& args,
      std::vector<Tensor>& feed);
  Status FillInputShapes(
    const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args,
    OpKernelContext* ctx,
    std::vector<XlaCompiler::Argument>* args,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info);

  Status CompileImpl(
                const XlaCompiler::Options& options,
                const NameAttrList& function,
                std::vector<XlaCompiler::Argument> args,
                const std::function<Status(
                        XlaCompiler* compiler,
                        XlaCompiler::CompilationResult*,
                        const XlaCompiler::CompileOptions& compile_options,
                        const NameAttrList& function,
                        absl::Span<const XlaCompiler::Argument> args,
                        std::shared_ptr<InputsShapeInfo> inputs_shape_info)>& compile_fn,
                const XlaCompiler::CompileOptions& compile_options,
                absl::optional<int64> compile_threshold,
                std::shared_ptr<InputsShapeInfo> inputs_shape_info);

  Status Warmup(
    const XlaCompiler::Options& options, const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args,
    const XlaCompiler::CompileOptions& compile_options,
    const std::function<Status(XlaCompiler* compiler,
            XlaCompiler::CompilationResult*,
            const XlaCompiler::CompileOptions& compile_options,
            const NameAttrList& function,
            absl::Span<const XlaCompiler::Argument> args,
            std::shared_ptr<InputsShapeInfo> inputs_shape_info)>& compile_fn,
    absl::optional<int64> compile_threshold,
    std::shared_ptr<InputsShapeInfo> inputs_shape_info);

 void FillShapeInferCtx(std::shared_ptr<InputsShapeInfo> inputs_shape_info,
     std::string device) {
   if (device.find("CPU") != device.npos || device.find("cpu") != device.npos) {
      inputs_shape_info->is_cpu_device = true;
   } else {
      inputs_shape_info->is_cpu_device = false;
   }

    inputs_shape_info->graph_properties = graph_properties_;
  }

  Status GetPaddingCachedShape(
      std::shared_ptr<InputsShapeInfo> inputs_shape_info);

private:
  const int MIN_PAD_VAL = 8;
  std::shared_ptr<XlaArgumentDumper> arg_dumper_;
  std::string name_;
  uint64 graph_key_ = 0;
  XlaCompilationCache* cache_;
  mutex graph_properties_mu_;
  // Record xla compiled shapes
  mutex cached_input_shapes_mu_;
  std::vector<std::shared_ptr<InputsShapeInfo>> cached_inputs_shapes_;
  
  std::shared_ptr<grappler::GraphProperties> graph_properties_;
  void InitShapeInferEntity(
      std::shared_ptr<InputsShapeInfo> inputs_shape_info);

  inline std::shared_ptr<InputsShapeInfo>
        PaddingInputs(std::shared_ptr<InputsShapeInfo> inputs);
  inline bool CheckConstEqual(
      const std::shared_ptr<InputsShapeInfo> cached_inputs,
      const std::shared_ptr<InputsShapeInfo> inputs);
  inline Status ValidateCluster(
      const std::vector<int>& base_dims,
      std::shared_ptr<InputsShapeInfo> inputs_shape_info);
  inline void pad_to_power(int& val, const std::vector<std::vector<int>>& shape_rules);
  mutex warmup_mu_;
  bool has_warmup_ = false;
  Status SortCaches();
};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_COMPILATION_CACHE_H_
