#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_XLA_PADDING_RULE_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_XLA_PADDING_RULE_H_

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/mutex.h"
#include <vector>

namespace tensorflow {
using namespace shape_inference;

namespace xla_padding_rule{

enum PaddingState { INVALID=-1, UNKNOWN=0, VALID=1 };
class XlaPaddingRule {
public:
  XlaPaddingRule();
  PaddingState GetGraphPaddingValid(const std::string& inputs_signature="default");
  void SetGraphPaddingValid(const std::string& inputs_signature, PaddingState state);
  // Only for unit test
  void ForceRestPaddingValid();

  void CheckNodeIsPaddingValid(const NodeDef& node, InferenceContext* ic, bool is_fast_mode);

  static bool IsWhiteListOp(std::string op);
  static bool IsBlackListOp(std::string op);
  static bool IsShapeSensitiveOp(std::string op);
private:
  bool ValidateByOp(const NodeDef& node, const std::vector<std::vector<int>>& diff_dims, InferenceContext* ic);

  // We could say,         the cluster is padding invalid
  // but we could not say, the cluster is padding valid
  // We could only say,    the cluster is padding valid in condition of one input shape

  // key: input difference signature, e.g:"001_01_00"
  // value: state. 0 unknowned; -1 Invalid; 1 valid
  std::map<std::string, PaddingState> cluster_validate_state_;
  // global flag for cluster_validate_state_ed
  bool cluster_validate_state_failed_ = false;
  mutex cluster_validate_state_mu_;
  std::vector<TensorShape> SaveNodeInputShape(const NodeDef* node, InferenceContext* ic, bool is_fast_mode);
  std::map<std::string, std::vector<TensorShape>> node_input_shapes_;

};
}  // xla_padding_rule
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_XLA_PADDING_RULE_H_
