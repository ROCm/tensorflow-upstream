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

#include "mlir/Support/TypeID.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include "llvm/Support/TypeName.h"

#if 0
namespace mlir {

class TypeID {
  /// This class represents the storage of a type info object.
  /// Note: We specify an explicit alignment here to allow use with
  /// PointerIntPair and other utilities/data structures that require a known
  /// pointer alignment.
  struct alignas(8) Storage {};

public:
  TypeID() : TypeID(get<void>()) {}

  /// Comparison operations.
  inline bool operator==(const TypeID &other) const {
    return storage == other.storage;
  }
  inline bool operator!=(const TypeID &other) const {
    return !(*this == other);
  }

  /// Construct a type info object for the given type T.
  template <typename T>
  static TypeID get();
  template <template <typename> class Trait>
  static TypeID get();

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(storage);
  }
  static TypeID getFromOpaquePointer(const void *pointer) {
    return TypeID(reinterpret_cast<const Storage *>(pointer));
  }

  /// Enable hashing TypeID.
  friend ::llvm::hash_code hash_value(TypeID id);

private:
  TypeID(const Storage *storage) : storage(storage) {}

  /// The storage of this type info object.
  const Storage *storage;

  friend class TypeIDAllocator;
};

class FallbackTypeIDResolver {
protected:
  /// Register an implicit type ID for the given type name.
  static TypeID registerImplicitTypeID(StringRef name);
};

template <typename T, typename Enable = void>
class TypeIDResolver : public FallbackTypeIDResolver {
public:
  /// Trait to check if `U` is fully resolved. We use this to verify that `T` is
  /// fully resolved when trying to resolve a TypeID. We don't technically need
  /// to have the full definition of `T` for the fallback, but it does help
  /// prevent situations where a forward declared type uses this fallback even
  /// though there is a strong definition for the TypeID in the location where
  /// `T` is defined.
  template <typename U>
  using is_fully_resolved_trait = decltype(sizeof(U));
  template <typename U>
  using is_fully_resolved = llvm::is_detected<is_fully_resolved_trait, U>;

  static TypeID resolveTypeID() {
    static_assert(is_fully_resolved<T>::value,
                  "TypeID::get<> requires the complete definition of `T`");
    static TypeID id = registerImplicitTypeID(llvm::getTypeName<T>());
    return id;
  }
};
};
#endif
/*
namespace mlir {
template <typename T, typename Enable = void>
    class TypeIDResolver;
};
*/
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"

namespace tf_device {

TensorFlowDeviceDialect::TensorFlowDeviceDialect(mlir::MLIRContext *context)
    : Dialect(/*name=*/"tf_device", context, mlir::TypeID::get<TensorFlowDeviceDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

}  // namespace tf_device

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.cc.inc"



