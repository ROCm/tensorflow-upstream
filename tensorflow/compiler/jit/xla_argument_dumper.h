#ifndef TENSORFLOW_COMPILER_JIT_XLA_ARGUMENT_DUMPER_H_
#define TENSORFLOW_COMPILER_JIT_XLA_ARGUMENT_DUMPER_H_

#include <atomic>
#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/compiler/jit/xla_argument.pb.h"

namespace tensorflow {

// The XlaArgumentDumper class dump the compiled arguments.
// which will be used to warmup when restart.
//
class XlaArgumentDumper {
public:
  XlaArgumentDumper();
  
  // Dumps the arguments to local file
  Status DumpXlaArguments(const std::vector<XlaCompiler::Argument>& args,
        const uint64 graph_key,
        const string& uuid);

  // Parse arguments from local file
  Status ParseFromFile(
        const std::shared_ptr<InputsShapeInfo>& base,
        const uint64 graph_key,
        std::vector<std::vector<XlaCompiler::Argument>>& args_array,
        std::vector<std::shared_ptr<InputsShapeInfo>>& inputs_shape_info_array);

private:
  // convert xla arguments to proto
  bool AsProto(const std::vector<XlaCompiler::Argument>& args,
        XlaArgumensProto& protos);
  // convert proto to xla arguments
  bool FromProto(const XlaArgumensProto& protos,
       std::vector<XlaCompiler::Argument>& args);

  std::shared_ptr<InputsShapeInfo> BuildInputsShapeInfo(
       const std::shared_ptr<InputsShapeInfo>& base,
       const std::vector<XlaCompiler::Argument>& args);

  string cache_dir_; 

};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_ARGUMENT_DUMPER_H_
