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

// This is the definition file for the TensorFlow Device Dialect.

#ifdef TF_DEVICE_DIALECT
#else
#define TF_DEVICE_DIALECT

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// TensorFlow Device Dialect definitions
//===----------------------------------------------------------------------===//

def TfDevice_Dialect : Dialect {
  let name = "tf_device";

  let description = [{
    The TensorFlow Device dialect.

    This dialect contains operations to describe/launch computations on devices.
    These operations do not map 1-1 to TensorFlow ops and requires a lowering
    pass later to transform them into Compile/Run op pairs, like XlaCompile and
    XlaRun.
}];

  let cppNamespace = "::mlir::tf_device";
}

//===----------------------------------------------------------------------===//
// TensorFlow Device Dialect Ops definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class TfDevice_Op<string mnemonic, list<Trait> traits = []> :
    Op<TfDevice_Dialect, mnemonic, traits> { }

def TfDevice_LaunchOp : TfDevice_Op<"launch",
                                    [SingleBlockImplicitTerminator<"ReturnOp">]>
{
  let summary = [{The `tf_device.launch` op captures all needed live-in values
                  and launches containing operations on target device.}];

  let arguments = (ins
    StrAttr:$device
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    bool WrapsSingleOp();
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$device, "TypeRange":$result_types),
    [{
      $_state.addAttribute("device", device);
      $_state.addTypes(result_types);
      $_state.addRegion();
    }]>
  ];

  let hasCanonicalizer = 1;
}

def TfDevice_ReturnOp : TfDevice_Op<"return",
                                    [Terminator, HasParent<"LaunchOp">]> {
  let summary = [{
    The `tf_device.return` operation terminates and returns values from
    `tf_device.launch` operation;
  }];

  let arguments = (ins
    Variadic<AnyType>:$results
  );

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, {});
    }]>
   ];

  // let verifier = ?;
}

def TfDevice_LaunchFuncOp : TfDevice_Op<"launch_func", []> {
  let summary = [{
    The `tf_device.launch_func` launches a function on target device.
  }];

  let arguments = (ins
    StrAttr:$device,
    SymbolRefAttr:$func,
    Variadic<AnyType>:$operands);

  let results = (outs
    Variadic<AnyType>:$results
  );
}

def TfDevice_ClusterOp : TfDevice_Op<"cluster",
    [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = [{
The `tf_device.cluster` op wraps containing operations in a region.
  }];

  let description = [{
This op can be used to group operations, and captures all needed live-in values.

Optional policy attribute allows to tag clusters with a policy name that was
used to form the cluster.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$policy);

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes),
    [{
      build($_builder, $_state, resultTypes, mlir::StringAttr {});
    }]>
  ];

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
  }];

  let hasCanonicalizer = 1;
}

def TfDevice_ClusterFuncOp : TfDevice_Op<"cluster_func",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = [{
The `tf_device.cluster_func` launches a function containing the body of a
cluster.
  }];

  let description = [{
This op is used for outlining a cluster.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$func,
    Variadic<AnyType>:$args
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return getArgs(); }
    MutableOperandRange getArgOperandsMutable() {
      return getArgsMutable();
    }
    // returns the function that this operation will launch.
    func::FuncOp getFuncOp() {
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getFuncAttr());
    }
    CallInterfaceCallable getCallableForCallee() {
      return getFuncAttr();
    }
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee);
  }];
}

#endif // TF_DEVICE_DIALECT
