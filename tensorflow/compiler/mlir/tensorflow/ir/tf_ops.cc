/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/FoldInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/dialect.h.inc"

#include "tensorflow/compiler/mlir/tensorflow/ir/dialect.cc.inc"

namespace mlir {
namespace TF {

// This all moved under tensorflow/core/ir/types and these using declaration are
// to help with the transition.
using ::mlir::tf_type::FuncAttr;         // NOLINT
using ::mlir::tf_type::PlaceholderAttr;  // NOLINT
using ::mlir::tf_type::ShapeAttr;        // NOLINT
using ::mlir::tf_type::TensorProtoAttr;  // NOLINT

}  // end namespace TF
}  // end namespace mlir

//#include "tensorflow/core/common_runtime/inline_function_utils.h"
//#include "tensorflow/core/common_runtime/lower_function_call_inline_policy.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/tensor_format.h"


//#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
namespace ResourceEffects {

struct Variable : ::mlir::SideEffects::Resource::Base<Variable> {
  StringRef getName() final { return "Variable"; }
};

struct Stack : ::mlir::SideEffects::Resource::Base<Stack> {
  StringRef getName() final { return "Stack"; }
};

struct TensorArray : ::mlir::SideEffects::Resource::Base<TensorArray> {
  StringRef getName() final { return "TensorArray"; }
};

struct Summary : ::mlir::SideEffects::Resource::Base<Summary> {
  StringRef getName() final { return "Summary"; }
};

struct LookupTable : ::mlir::SideEffects::Resource::Base<LookupTable> {
  StringRef getName() final { return "LookupTable"; }
};

struct DatasetSeedGenerator
    : ::mlir::SideEffects::Resource::Base<DatasetSeedGenerator> {
  StringRef getName() final { return "DatasetSeedGenerator"; }
};

struct DatasetMemoryCache
    : ::mlir::SideEffects::Resource::Base<DatasetMemoryCache> {
  StringRef getName() final { return "DatasetMemoryCache"; }
};

struct DatasetIterator : ::mlir::SideEffects::Resource::Base<DatasetIterator> {
  StringRef getName() final { return "DatasetIterator"; }
};

// Special resource type to track TPU Embedding specific ops, which must execute
// but do not have side effects with one another or with resource variable ops.
struct TPUEmbedding : ::mlir::SideEffects::Resource::Base<TPUEmbedding> {
  StringRef getName() final { return "TPUEmbedding"; }
};

// Resource corresponding to GeneratorOp.
struct GeneratorOp : public ::mlir::SideEffects::Resource::Base<GeneratorOp> {
  StringRef getName() final { return "Generator"; }
};

struct Send : public ::mlir::SideEffects::Resource::Base<Send> {
  StringRef getName() final { return "Send"; }
};

struct Recv : public ::mlir::SideEffects::Resource::Base<Recv> {
  StringRef getName() final { return "Recv"; }
};

struct XlaHostCompute
    : public ::mlir::SideEffects::Resource::Base<XlaHostCompute> {
  StringRef getName() final { return "XlaHostCompute"; }
};

struct RandomGenerator
    : public ::mlir::SideEffects::Resource::Base<RandomGenerator> {
  StringRef getName() final { return "RandomGenerator"; }
};

struct TPUExecute : public ::mlir::SideEffects::Resource::Base<TPUExecute> {
  StringRef getName() final { return "TPUExecute"; }
};

struct MustExecute : public ::mlir::SideEffects::Resource::Base<MustExecute> {
  StringRef getName() final { return "MustExecute"; }
};

struct CollectiveReduceOrdering
    : public ::mlir::SideEffects::Resource::Base<CollectiveReduceOrdering> {
  StringRef getName() final { return "CollectiveReduceOrdering"; }
};

struct NcclAllReduceOrdering
    : public ::mlir::SideEffects::Resource::Base<NcclAllReduceOrdering> {
  StringRef getName() final { return "NcclAllReduceOrdering"; }
};

struct GlobalIterId : public ::mlir::SideEffects::Resource::Base<GlobalIterId> {
  StringRef getName() final { return "GlobalIterId"; }
};

struct XlaLaunch : public ::mlir::SideEffects::Resource::Base<XlaLaunch> {
  StringRef getName() final { return "XlaLaunch"; }
};

struct WriteTrainingPredictions
    : public ::mlir::SideEffects::Resource::Base<WriteTrainingPredictions> {
  StringRef getName() final { return "WriteTrainingPredictions"; }
};

struct _XlaRun : public ::mlir::SideEffects::Resource::Base<_XlaRun> {
  StringRef getName() final { return "_XlaRun"; }
};

// Returns true iff resource type with given ID is only self-dependent, i.e.,
// there are no dependencies to other resource types (including unknown resource
// type).
inline bool IsOnlySelfDependent(TypeID resource_type_id) {
  return resource_type_id == ResourceEffects::Send::getResourceID() ||
         resource_type_id == ResourceEffects::Recv::getResourceID();
}

}  // namespace ResourceEffects

#include <optional>

#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
//#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/util/device_name_utils.h"


namespace tensorflow {
static constexpr const char* const kNoInlineAttr = "_noinline";

// 2.x lower_function_call_inline_policy.h

#if 0

// LINT.IfChange
enum class FunctionCallInlinePolicy {
  // Place input nodes on the same device as the corresponding caller input
  // node. Do not specify any placement for all other nodes.
  kDefaultPlacer,

  // Place all nodes on the same device as caller node.
  kSingleDevicePlacer,

  // Place input nodes on the same device as the corresponding caller input
  // node. Do not place output node. Place control nodes on the same device as
  // caller node. For all function body nodes overrides job, replica and task
  // parts of the device assignment to match function caller node.
  kMultiDevicePlacer
};

// LINT.ThenChange(inline_function_utils.h,\
//   ../../compiler/mlir/tensorflow/ir/tf_ops.cc)

FunctionCallInlinePolicy GetFunctionCallInlinePolicy(
    bool is_partioned_call, bool has_lower_as_multi_device_function_attr) {
  if (is_partioned_call || has_lower_as_multi_device_function_attr)
    return FunctionCallInlinePolicy::kMultiDevicePlacer;
  return FunctionCallInlinePolicy::kSingleDevicePlacer;
}

FunctionCallInlinePolicy GetFunctionCallInlinePolicy(const Node* n) {
  bool match;
  bool found = TryGetNodeAttr(
      n->attrs(), LowerFunctionalOpsConstants::kLowerAsMultiDeviceFunctionAttr,
      &match);
  return GetFunctionCallInlinePolicy(n->IsPartitionedCall(), found && match);
}

#endif

struct LowerFunctionalOpsConstants {
  static constexpr const char* const kLowerUsingSwitchMergeAttr =
      "_lower_using_switch_merge";
  static constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
      "_lower_as_multi_device_function";
};
};

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// TF Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct TFConstantFoldInterface : public DialectFoldInterface {
  TFConstantFoldInterface(Dialect *dialect) : DialectFoldInterface(dialect) {}
  LogicalResult fold(Operation *op, ArrayRef<Attribute> operands,
                     SmallVectorImpl<OpFoldResult> &results) const final {
    return TensorFlowDialect::constantFold(op, operands, results);
  }
};

using DeviceNameUtils = tensorflow::DeviceNameUtils;

void MergeUnsetDevNames(DeviceNameUtils::ParsedName* target,
                                         const DeviceNameUtils::ParsedName& other) {
  if (other.has_job && !target->has_job) {
    target->has_job = other.has_job;
    target->job = other.job;
  }

  if (other.has_replica && !target->has_replica) {
    target->has_replica = other.has_replica;
    target->replica = other.replica;
  }

  if (other.has_task && !target->has_task) {
    target->has_task = other.has_task;
    target->task = other.task;
  }

  if (other.has_type && !target->has_type) {
    target->has_type = other.has_type;
    target->type = other.type;
  }

  if (other.has_id && !target->has_id) {
    target->has_id = other.has_id;
    target->id = other.id;
  }
}


// Helper function that implements the multi-device inlining policy behavior
// for the inliner hook. In particular, for all function body nodes set unset
// placement attributes to match the function call node.
void MultiDeviceProcessInlinedCallBlocks(
    Operation *call, iterator_range<Region::iterator> inlinedBlocks) {
  // Duplicate of the logic in MultiDeviceFunctionBodyPlacer::BodyNodeDevice
  // LINT.IfChange
  auto device_id = StringAttr::get(call->getContext(), "device");
  auto caller_device = call->getAttrOfType<StringAttr>(device_id);
  if (!caller_device) return;

  DeviceNameUtils::ParsedName caller_parsed_device;
  if (!DeviceNameUtils::ParseFullName(caller_device.getValue().str(),
                                      &caller_parsed_device))
    return;

  MLIRContext *context = call->getContext();
  auto node_device = [&](Operation *n) -> StringAttr {
    auto device = n->getAttrOfType<StringAttr>(device_id);
    if (!device || device.getValue().empty()) return caller_device;

    DeviceNameUtils::ParsedName ndef_parsed_device;
    if (!DeviceNameUtils::ParseFullName(device.getValue().str(),
                                        &ndef_parsed_device))
      return device;
    MergeUnsetDevNames(&ndef_parsed_device,
                                        caller_parsed_device);
    return StringAttr::get(
        context, DeviceNameUtils::ParsedNameToString(ndef_parsed_device));
  };
  // LINT.ThenChange(../../../../core/common_runtime/inline_function_utils.cc)

  for (Block &block : inlinedBlocks) {
    block.walk([&](Operation *op) {
      if (op->getDialect() == call->getDialect())
        op->setAttr(device_id, node_device(op));
    });
  }
}

struct TFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Returns if it's legal to inline 'callable' into the 'call', where 'call' is
  // a TF operation.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Skip inlining for TPUPartitionedCalls and RemoteCalls.
#if 0    
    if (isa<TPUPartitionedCallOp>(call)) return false;
#endif    
    if (isa<RemoteCallOp>(call)) return false;
    // Maintain inlining for  `tf.function`s with jit_compile option.
    if (callable->hasAttr("tf._XlaMustCompile")) return true;
    auto noinline_attr_name = absl::StrCat("tf.", tensorflow::kNoInlineAttr);
    if (auto noinline_attr =
            callable->getAttrOfType<BoolAttr>(noinline_attr_name))
      return !noinline_attr.getValue();
    return true;
  }

  // Returns if its legal to inline 'src' region into the 'dest' region
  // attached to a TF operation.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Allow inlining in regions attached to region based control flow
    // operations only if the src region is a single block region
    return isa<IfRegionOp, CaseRegionOp, WhileRegionOp>(dest->getParentOp()) &&
           llvm::hasSingleElement(*src);
  }

  // Returns true if its legal to inline a TF operation `op` into the `dest`
  // region.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &) const final {
    // An op is legal to inline if either of the following conditions is true:
    // (a) Its legal to duplicate the Op.
    // (b) The Op is inside a single use function. If that function is inlined,
    //     post inlining, the function will be dead and eliminated from the IR.
    //     So there won't be any code duplication.
    // plus the function caller op can be replaced by inlined ops.
    return !wouldBeCloned || TensorFlowDialect::CanDuplicate(op);
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
    if (!result_type.isa<TensorType>() || !input.getType().isa<TensorType>())
      return nullptr;
    return builder.create<TF::CastOp>(conversion_loc, result_type, input,
                                      /*truncate=*/builder.getBoolAttr(false));
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const final {
// FIXME
/*
    bool has_lower_as_multi_device_function_attr = false;
    if (auto lower = call->getAttrOfType<BoolAttr>(
            tensorflow::LowerFunctionalOpsConstants::
                kLowerAsMultiDeviceFunctionAttr))
      has_lower_as_multi_device_function_attr = lower.getValue();
    tensorflow::FunctionCallInlinePolicy policy =
        tensorflow::GetFunctionCallInlinePolicy(
            isa<PartitionedCallOp, StatefulPartitionedCallOp>(call),
            has_lower_as_multi_device_function_attr);

    if (policy == tensorflow::FunctionCallInlinePolicy::kMultiDevicePlacer)
      return MultiDeviceProcessInlinedCallBlocks(call, inlinedBlocks);
*/    
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// TF Dialect
//===----------------------------------------------------------------------===//

// Returns true if the op can be duplicated.
bool TensorFlowDialect::CanDuplicate(Operation *op) {
  // If the op is marked with the cannot duplicate trait, it cannot be
  // duplicated.
  if (op->hasTrait<OpTrait::TF::CannotDuplicate>()) return false;

  // If the op has no memory side effects, it can be duplicated.
  if (isMemoryEffectFree(op)) return true;

  // If the op is marked stateless using the `is_stateless` attribute, that
  // attribute determines if the op can be duplicated.
  if (auto is_stateless = op->getAttrOfType<BoolAttr>("is_stateless"))
    return is_stateless.getValue();

  // Assume ops can be duplicated if modelled.
  return op->isRegistered();
}

// TF dialect fallback for MemoryEffectOpInterface. The filtering for returning
// the interface is done in the return below and here it is empty as it is only
// returned for known not-stateful and unmodelled ops.
struct TensorFlowRegistryEffectInterfaceFallback
    : public MemoryEffectOpInterface::FallbackModel<
          TensorFlowRegistryEffectInterfaceFallback> {
  static bool classof(Operation *op) { return true; }
  void getEffects(
      Operation *op,
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
          &effects) const {}
};

void *TensorFlowDialect::getRegisteredInterfaceForOp(
    mlir::TypeID interface, mlir::OperationName opName) {
  if (interface == TypeID::get<mlir::MemoryEffectOpInterface>()) {
    // Don't use fallback for modelled ops.
    if (opName.isRegistered()) return nullptr;

    // Only use fallback interface for known not-stateful ops.
    const tensorflow::OpRegistrationData *op_reg_data = nullptr;
    tensorflow::Status s = tensorflow::OpRegistry::Global()->LookUp(
        opName.stripDialect().str(), &op_reg_data);
    return (s.ok() && !op_reg_data->op_def.is_stateful())
               ? fallback_effect_op_interface_
               : nullptr;
  }

  return nullptr;
}

// Returns true if the op can have side effects.
bool TensorFlowDialect::CanHaveSideEffects(Operation *op) {
  // If the op has no memory side effects, it has no side effects
  if (isMemoryEffectFree(op)) return false;

  // If the op is marked stateless using the `is_stateless` attribute, then
  // it has no side effects.
  if (auto is_stateless = op->getAttrOfType<BoolAttr>("is_stateless"))
    return !is_stateless.getValue();

  // Terminators defined in the TF dialect do not have side effects.
  if (op->hasTrait<OpTrait::IsTerminator>()) return false;

  // Otherwise assume that the op can have side effects.
  return true;
}

// Hook functions which may add additional operations to the dialect.
// These are invoked at construction time.
static DenseMap<TypeID, TensorFlowDialect::AdditionalOpFunction>
    &GetAdditionalOperationHooks() {
  static auto *additional_operation_hooks =
      new DenseMap<TypeID, TensorFlowDialect::AdditionalOpFunction>();
  return *additional_operation_hooks;
}

void TensorFlowDialect::RegisterAdditionalOperationHook(
    TypeID id, AdditionalOpFunction fn) {
  GetAdditionalOperationHooks().try_emplace(id, std::move(fn));
}

TensorFlowDialect::ConstantFoldHook TensorFlowDialect::constant_fold_hook_;

TensorFlowDialect::TensorFlowDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf", context, TypeID::get<TensorFlowDialect>()) {
  context->getOrLoadDialect<::mlir::tf_type::TFTypeDialect>();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc.inc"
      >();
  addInterfaces<TFInlinerInterface, TFConstantFoldInterface>();
  fallback_effect_op_interface_ =
      new TensorFlowRegistryEffectInterfaceFallback();

  // Support unknown operations because not all TensorFlow operations are
  // registered.
  allowUnknownOperations();

  for (auto &hook : GetAdditionalOperationHooks()) {
    hook.second(*this);
  }
}

TensorFlowDialect::~TensorFlowDialect() {
  delete fallback_effect_op_interface_;
}

Type TensorFlowDialect::parseType(DialectAsmParser &parser) const {
  StringRef spec = parser.getFullSymbolSpec();
  llvm::SMLoc loc = parser.getCurrentLocation();
  parser.emitError(
      loc, "tf dialect has no types, potentially meant !tf_type." + spec);
  return nullptr;
}

Attribute TensorFlowDialect::parseAttribute(DialectAsmParser &parser,
                                            Type type) const {
  StringRef spec = parser.getFullSymbolSpec();
  llvm::SMLoc loc = parser.getCurrentLocation();
  parser.emitError(
      loc, "tf dialect has no attributes, potentially meant #tf_type." + spec);
  return nullptr;
}

Operation *TensorFlowDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return builder.create<ConstOp>(loc, type, value);
}

}  // namespace TF
}  // namespace mlir
