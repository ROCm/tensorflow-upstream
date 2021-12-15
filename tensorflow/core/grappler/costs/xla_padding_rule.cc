
#include "tensorflow/core/grappler/costs/xla_padding_rule.h"

#include <set>

namespace tensorflow {
namespace xla_padding_rule {

namespace {
  // Ops in white lists are ignored when checking
  // Since we assume they donot affect data layout
  const std::set<std::string> white_list_ops = {
    "Cast", "Bitcast", "Identity", "IdentityN", "_HostCast", "_Arg", "Const", "_Retval",
    "Add", "AddN", "AddV2", "Mul", "Sub", "Div", "DivNoNan", "Log","Exp",
    "TruncateDiv","RealDiv","SquaredDifference",
    "Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh", "Atan2", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", 
    "Round", "Square", "Sqrt", "Rsqrt", "Sum", "Tan", "Tanh", "Abs", "Round", "Equal", "NotEqual", "ApproximateEqual",
    "ZerosLike", "OnesLike", "Greater","GreaterEqual","Less", "LessEqual","LogicalAnd","LogicalNot","LogicalOr",
    "Ceil", "Floor", "FloorDiv", "Sign", "Rint","SquareDifference", "BiasAdd", "BiasAddV1",
    "Minimum", "Maximum", "Min","Max","Neg","Mean","Mod", "FloorMod", "TruncateMod", "Pow",
    "BitwiseAnd", "BitwiseOr", "BitwiseXor", "TruncatedNormal", 
    "Transpose", "Bucketize", "Fill", 
    "Pack", "Stack",  "ExpandDims",
    "StopGradient", "PreventGradient", 
    "Assert", "Case", 
    "LeftShift", "Log1p", "LogicalAnd", "LogicalNot", "LogicalOr", "LogicalOr",
    "Mod", "MulNoNan", "Multinomial", "Neg", "NoOp", "Pad", "PadV2",  "Real", "RealDiv", 
    "Sigmoid", "Selu", "LeakyRelu", "Relu", "Relu6"
  };
  const std::set<std::string> shape_ops = {"Shape", "ShapeN", "Rank","Size", "TensorArraySizeV3"};
  const std::set<std::string> black_list_ops = {
    "If",  "While", "GatherV2", "Gather", "GatherNd", "Select", "SelectV2", 
    // TODO
    "All", "Any", "ArgMax", "ArgMin", "AvgPool", "AvgPool3D", "BroadcastArgs", "BroadcastGradientArgs",
    "BroadcastTo", "Complex", "ComplexAbs", "ConcatOffset", "Cumprod", "Cumsum", "Diag", "Elu", "Empty",
    "Erf", "Erfc", "Expm1",  "FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3", "Inv", "Invert",
    "InvertPermutation", "IsFinite", "IsInf", "IsNan", "Lgamma", "LogSoftmax", "MaxPool", "MaxPool3D", 
    "MaxPoolV2",   "Range", "ResizeBilinear", "ResizeNearestNeighbor", "Reverse", "ReverseSequence", 
    "ReverseV2", "RightShift", "SoftmaxCrossEntropyWithLogits", "TopKV2", "Squeeze", 

    // unknown ops
    "AdjustContrastv2", "AdjustHue", "AdjustSaturation", "AssignAddVariableOp","AssignSubVariableOp",
    "AssignVariableOp", "BatchToSpace", "BatchToSpaceND", "Cholesky", "ClipByValue",
    "Conj", "ConjugateTranspose", "ControlTrigger", "Conv2D", "Conv2DBackpropFilter", "Conv2DBackpropInput",
    "Conv3D", "Conv3DBackpropFilterV2", "Conv3DBackpropInputV2", "Cross", "DataFormatDimMap", "DataFormatVecPermute",
    "DepthToSpace", "DepthwiseConv2dNative", "DepthwiseConv2dNativeBackpropFilter", "DepthwiseConv2dNativeBackpropInput", 
    "DiagPart", "Digamma", "DynamicStitch", "EluGrad", "EmptyTensorList", "ExtractImagePatches", "FFT", "FFT2D", "FFT3D",
    "FakeParam", "FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxArgsGradient", "FakeQuantWithMinMaxVars", "FakeQuantWithMinMaxVarsGradient",
    "HSVToRGB", "IFFT", "IFFT2D", "IFFT3D", "IRFFT", "IRFFT2D", "IRFFT3D", "Imag", "InTopKV2", "L2Loss",
    "LRN", "LinSpace", "ListDiff",
    "MatrixBandPart", "MatrixDiag", "MatrixDiagPart", "MatrixDiagPartV2", "MatrixDiagV2", "MatrixInverse", 
    "MatrixSetDiag", "MatrixSetDiagV2", "MatrixTriangularSolve", "MirrorPad",  "NextAfter", "NonMaxSuppressionV4", 
    "OneHot", "ParallelDynamicStitch", "ParameterizedTruncatedNormal",  "PlaceholderWithDefault", 
    "Prod", "Qr", "QuantizeAndDequantizeV2", "QuantizeAndDequantizeV3", "RFFT", "RFFT2D", "RFFT3D", 
    "RGBToHSV", "RandomShuffle", "RandomStandardNormal", "RandomUniform", "RandomUniformInt", "ReadVariableOp",
    "Reciprocal", "ReciprocalGrad", "ResourceApplyAdaMax", "ResourceApplyAdadelta", "ResourceApplyAdagrad", 
    "ResourceApplyAdagradDA", "ResourceApplyAdagradV2", "ResourceApplyAdam", "ResourceApplyAddSign", 
    "ResourceApplyCenteredRMSProp", "ResourceApplyFtrl", "ResourceApplyFtrlV2", "ResourceApplyGradientDescent", 
    "ResourceApplyKerasMomentum", "ResourceApplyMomentum", "ResourceApplyPowerSign", "ResourceApplyProximalAdagrad", 
    "ResourceApplyProximalGradientDescent", "ResourceApplyRMSProp", "ResourceGather", "ResourceScatterAdd", 
    "ResourceScatterDiv", "ResourceScatterMax", "ResourceScatterMin", "ResourceScatterMul", "ResourceScatterNdAdd", 
    "ResourceScatterNdSub", "ResourceScatterNdUpdate", "ResourceScatterSub", "ResourceScatterUpdate", 
    "ResourceStridedSliceAssign", "Rint", "Roll", "RsqrtGrad", "ScatterNd", "SelfAdjointEigV2", 
    "Snapshot", "Softplus", "SoftplusGrad", "Softsign", "SoftsignGrad", "SpaceToBatch", "SpaceToBatchND",
    "SpaceToDepth", "SparseMatMul", "SparseSoftmaxCrossEntropyWithLogits", "SparseToDense", "SqrtGrad", "Square",
    "StackCloseV2", "StackPopV2", "StackPushV2", "StatefulPartitionedCall", 
    "StatefulStandardNormalV2", "StatefulTruncatedNormal", "StatefulUniform", "StatefulUniformFullInt", 
    "StatefulUniformInt", "StatelessIf", "StatelessMultinomial", "StatelessRandomNormal", "StatelessRandomUniform", 
    "StatelessRandomUniformInt", "StatelessTruncatedNormal", "StatelessWhile",  "StridedSliceGrad", "Svd", 
    "SymbolicGradient", "TanhGrad", "TensorArrayCloseV3", "TensorArrayConcatV3", "TensorArrayGatherV3", 
    "TensorArrayGradV3", "TensorArrayReadV3", "TensorArrayScatterV3", "TensorArraySplitV3", 
    "TensorArrayV3", "TensorArrayWriteV3", "TensorListElementShape", "TensorListFromTensor", "TensorListGather",
    "TensorListGetItem", "TensorListLength", "TensorListPopBack", "TensorListPushBack", "TensorListReserve", 
    "TensorListSetItem", "TensorListStack", "TensorScatterAdd", "TensorScatterSub", "TensorScatterUpdate", 
    "UnsortedSegmentMax", "UnsortedSegmentMin", "UnsortedSegmentProd", "UnsortedSegmentSum", "VarIsInitializedOp", 
    "VariableShape", "Xdivy", "XlaBroadcastHelper", "XlaConv", "XlaDequantize", "XlaDot", "XlaDynamicSlice", 
    "XlaDynamicUpdateSlice", "XlaEinsum", "XlaIf", "XlaKeyValueSort", "XlaPad", "XlaRecv", "XlaReduce", 
    "XlaReduceWindow", "XlaReplicaId", "XlaSelectAndScatter", "XlaSelfAdjointEig", "XlaSend", "XlaSort", 
    "XlaSvd", "XlaWhile", "Xlogy", "_ArrayToList", "_FusedBatchNormEx", "_ListToArray", "_UnaryOpsComposition",
    // Grad ops
    "BiasAddGrad", "AvgPool3DGrad", "AvgPoolGrad", "FusedBatchNormGrad", "FusedBatchNormGradV2", "FusedBatchNormGradV3",
    "LRNGrad", "LeakyReluGrad", "MaxPool3DGrad", "MaxPool3DGradGrad", "MaxPoolGrad", "MaxPoolGradGrad",
    "MaxPoolGradGradV2", "MaxPoolGradV2", "Relu6Grad", "ReluGrad", "ResizeBilinearGrad", "SeluGrad",  "SigmoidGrad"
  };

  bool in_vector(const std::vector<int>& vec, int val) {
    return find(vec.begin(), vec.end(), val) != vec.end();
  }

  string DebugString(const gtl::ArraySlice<int> arrays) {
    std::vector<string> vals;
    for (auto d : arrays) vals.push_back(std::to_string(d));
    return strings::StrCat("[", absl::StrJoin(vals, ","), "]");
  }

  /********************************************************/
  /********************************************************/
  /****            Add new Op rules Here *****************/
  /********************************************************/
  /********************************************************/
 
  inline bool ValidateSoftmax(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // softmax axis and pad dim can not be same!
    // tf.nn.softmax(
    //     logits,
    //     axis=None,
    //     name=None,
    //     dim=None
    // )

    // How does softmax work?
    // eg: inputs shape(2, 3)
    //   [[0, 1, 0],
    //    [1, 0, 1]]
    // if axis = 0, then output    
    // [[0.26894143 0.7310586  0.26894143]
    //  [0.7310586  0.26894143 0.7310586 ]]
    // if axis = 1, then output    
    // [[0.21194157 0.5761169  0.21194157]
    //  [0.42231882 0.15536241 0.42231882]]
    // default axis=-1, which means last dim
    // [[0.21194157 0.5761169  0.21194157]
    //  [0.42231882 0.15536241 0.42231882]]
    if (node.op() != "Softmax") return false;

    std::vector<int32> inputs_shape = InferenceContext::Dims(ic->input(0));
    int axis = inputs_shape.size() - 1;
    if (node.attr().count("axis") > 0) {
      axis = node.attr().at("axis").i();
      VLOG(1) << node.name() << "(" << node.op() << ") has axis attr, " << axis; 
    } else {
      VLOG(1) << node.name() << "(" << node.op() << ") has no axis attr, use last dim " << axis; 
    }

    CHECK(diff_dims.size() > 0); // This always true, because 
                                 // it will not be here if diff_dims.size() <= 0
    const auto& input_dims = diff_dims[0];
    for(auto dim: input_dims) {
      if (dim == axis) {
        LOG(WARNING) << "Validate " << node.name() << "(" << node.op() << ") xla auto padding rule failed;"
                     << " input shape " << DebugString(inputs_shape)
                     << " with padding dim=" << dim
                     << " and softmax axis=" << axis;
        return false;
      }
    }
    
    return true;
  }


  inline bool ValidateConcat(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // concat dim and pad dim can not be same!
    // input 0 shape (d00, d01, d02, d03 ... d0n)
    // input 1 shape (d10, d11, d12, d13 ... d1n)
    // If pad dim and concat dim are both 0
    // and pad value is m
    // Therefore, the output dim is 
    // (d00 + m + d10 + m, d01, d02, d03 ... d0n)
    // When slice to d00+d10 along dim 0
    // The output will not be correct.    

    // How does concat works ?
    // eg: input0 shape (2, 3), input1 shape (2, 3)
    // if concat dim is 0, then output (4, 3)
    // if concat dim is 1, then output (2, 6)

    if (node.op() != "ConcatV2" && node.op() != "Concat") return false;
    const Tensor* concat_dim_t = ic->input_tensor(ic->num_inputs() - 1);
    const int32 concat_dim = concat_dim_t->scalar<int32>()();
    VLOG(1) << "Concat dim " << concat_dim;
    for (size_t i = 0; i < diff_dims.size(); i++) {
      const auto& input_dims = diff_dims[i];
      for (int dim: input_dims) {
        if (dim == concat_dim) {
          LOG(WARNING) << "Validate " << node.name() << "(" << node.op() << ") xla auto padding rule failed;"
                       << " input " << i << " padding dim=" << dim
                       << " and concat dim=" << concat_dim;
          return false;
        }
      }
    }
    return true;
  }

  inline bool ValidateUnpack(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // Just like Slice, unpack axis and the dynamic dim
    // cannot be same
    if (node.op() != "Unpack") return false;
 
    int axis = node.attr().at("axis").i();
    // split dim not the dynamic dim of input 1
    CHECK(diff_dims.size() > 0); // This always true, because 
                                 // it will not be here if diff_dims.size() <= 0
    const auto& input_dims = diff_dims[0];
    for (int dim: input_dims) {
      if (dim == axis) {
        LOG(WARNING) << "Validate " << node.name() << "(" << node.op() << ") xla auto padding rule failed; "
            << " input  padding dim=" << dim
            << " and unpack dim=" << axis;
        return false;
      }
    }
    return true;
  }

  inline bool ValidateSplit(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // Just like Concat, split axis and the dynamic dim
    // cannot be same
    
    // How does split works?
    // tf.split(
    //    value,
    //    num_or_size_splits,
    //    axis=0,
    //    num=None,
    //    name='split'
    // )
    // eg: input (20, 30, 40)
    // if num_or_size_splits=2 and axis=0, then output [(10, 30, 40), [10, 30, 40]
    // if num_or_size_splits=[10, 5, 25] and axis=2, then output 
    // [(10, 30, 10), (10, 30, 5), (10, 30, 25)]
    
    if (node.op() != "Split" && node.op() != "SplitV") return false;
 
    const Tensor* split_dim_t = ic->input_tensor(0);
    const int32 split_dim = split_dim_t->scalar<int32>()();
    // split dim not the dynamic dim of input 1
    CHECK(diff_dims.size() > 0); // This always true, because 
                                  // it will not be here if diff_dims.size() <= 0
    const auto& input_dims = diff_dims[1];
    for (int dim: input_dims) {
      if (dim == split_dim) {
        LOG(WARNING) << "Validate " << node.name() << "(" << node.op() << ") xla auto padding rule failed; "
            << " input  padding dim=" << dim
            << " and split dim=" << split_dim;
        return false;
      }
    }
    return true;
  }

  /// DEPRECATED. 
  /// Pack and Stack add to white list
  inline bool ValidateStack(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // Stack is supported, becase they dont fuse 
    // datas together, but keeps datas along their dims

    // How does stack works?
    // eg: input0 (2, 3), input1(2, 3)
    // if stack dim is 0, then output (2, 2, 3)
    // if stack dim is 1, then output (2, 3, 2)
    if (node.op() != "Stack" && node.op() != "StackV2" && node.op() != "Pack") return false;

    if (node.attr().count("axis") <= 0) {
      LOG(WARNING) << node.name() << "(" << node.op() << ") has no axis attr";
      return false;
    }

    int pack_dim = node.attr().at("axis").i();
    VLOG(1) << "Pack dim " << pack_dim;
    return true;
  }

  inline bool ValidateReshape(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // We consider two shapes: the input shape
    // and the target shape.
    // We support only one dim dynamic both for 
    // input shape and target shape, and the dynamic
    // dim is dx and dy, respectivly.
    // Then the cumprod before dx and dy must be same,
    // the cumprod after dx and dy must be same, either.

    // eg: input dim is (32, 3, batch, 2, 5)
    // reshape value is (16, 2, 3, -1, 10)
    // the dynamic dim for input and target shape is d2 and d3
    // If 32 * 3 == 16 * 2 * 3 and 2 * 5 == 10
    // then the padding is valid

    if (node.op() != "Reshape") return false;
    CHECK(diff_dims.size() == 2) << node.op() << " must have 2 inputs but get " << diff_dims.size();
   
    // Input shape unchange, return
    if (diff_dims[0].size() == 0) return true;
    if (diff_dims[0].size() > 1) {
      LOG(WARNING) << node.name() << "(Reshape) only support one dynamic dim, but got " << diff_dims[0].size();
      return false;
    }
    
    // Get the mul of input dims before and after dynamic dim
    int input_dynamic_dim = diff_dims[0][0];
    std::vector<int32> inputs_shape = InferenceContext::Dims(ic->input(0));
    int cumprod_pre_dynamic_dim = 1, cumprod_post_dynamic_dim = 1;
    for (size_t i = 0; i < inputs_shape.size(); i++) {
      if (i < input_dynamic_dim) {
        cumprod_pre_dynamic_dim *= inputs_shape[i];
      } else if (i > input_dynamic_dim) {
        cumprod_post_dynamic_dim *= inputs_shape[i];
      }
    }

    // Get the mul of reshape dims before and after dynamic dim
    int shape_cumprod_pre_dynamic_dim = 1, shape_cumprod_post_dynamic_dim = 1;
    const Tensor* reshape_vals = ic->input_tensor(1);
    const int reshape_vals_count = InferenceContext::Value(ic->input(1), 0);
    const gtl::ArraySlice<int> reshape_array(reshape_vals->flat<int>().data(), reshape_vals_count);
    VLOG(1) << node.op() << " Dims " << DebugString(reshape_array);

    int unknown_dim = -1;
    for(size_t i = 0; i < reshape_array.size(); i++) {
      if (reshape_array[i] == -1) {
        if (unknown_dim != -1) {
          LOG(WARNING) << node.name() << "(Reshape) only support one dim unknown, but got 2";
          return false;
        }
        unknown_dim = i;
      }
    }
    // Input has dynamic batch, the shape cannot be static value
    CHECK(unknown_dim != -1);
    if (unknown_dim == 0) {
      VLOG(1) << "Reshape unknown_dim=0, succ";
      return true;
    } 
    for (size_t i = 0; i < reshape_array.size(); i++) {
      if (i < unknown_dim) {
        shape_cumprod_pre_dynamic_dim *= reshape_array[i];
      } else if (i > unknown_dim) {
        shape_cumprod_post_dynamic_dim *= reshape_array[i];
      }
    }
    
    // Compare the vals before and after dynamic dim
    if (cumprod_pre_dynamic_dim != shape_cumprod_pre_dynamic_dim ||
        cumprod_post_dynamic_dim != shape_cumprod_post_dynamic_dim) {
      LOG(WARNING) << node.name() << "(" << node.op() << ")  cannot compile"; 
      LOG(WARNING) << "cumprod_pre_dynamic_dim:" << cumprod_pre_dynamic_dim << " vs "  
                   << shape_cumprod_pre_dynamic_dim
                   << " cumprod_post_dynamic_dim:" << cumprod_post_dynamic_dim << " vs "
                    << shape_cumprod_post_dynamic_dim;
      return false;
    }
    return true;
  }

  inline bool ValidateTile(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // Tile behavies like Concat
    // Tile dim along the padding dim must be 1
    // Input shape is (d0, d1, d2 ... dn)
    // If pad dim is 0, pad value is m
    // And Tile shape is (2, 1, 1 ... 1)
    // The output shape will be (d0 + m + d0 + m, d1, d2 ... dn)
    // When slice to d0+d0 along dim 0
    // The output will be not correct.    

    // How does Tile works?
    // eg: input0 (2, 3), tile is (1, 2)
    // then output (2, 6)
    if (node.op() != "Tile") return false;

    const int input_dims = InferenceContext::Rank(ic->input(0));
    const Tensor* multiples = ic->input_tensor(1);
    const gtl::ArraySlice<int> multiples_array(multiples->flat<int>().data(), input_dims);
    VLOG(1) << node.op() << " Dims " << DebugString(multiples_array);

    for (int i = 0; i < diff_dims.size(); i++) {
      const auto& input_dims = diff_dims[i];
      for (int dim: input_dims) {
        if (multiples_array[dim] != 1) {
          LOG(WARNING) << "Validate " << node.name() << "(" << node.op() << ") xla auto padding rule failed; "
                       << "input " << i << " padding dim=" << dim
                       << "and Tile val=" << multiples_array[dim];
          return false;
        }
      }
    }
    return true;
  }

  inline bool ValidateSlice(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // Slice behavies like Tile
    // Slice piece along the padding dim must equal to the dim of input
    // That is, begin=0 and size=input shape

    // Input shape is (d0, d1, d2 ... dn)
    // If pad dim is 0, then
    // begin must be [0, x, x, x] and slice must be [d0, x, x, x]

    // How does Slice works?
    // tf.slice(inputs,begin,size,name='')
    // eg: input is
    //   [ 
    //      [1 2 3]
    //      [4 5 6] 
    //   ]
    // so the input shape is (2, 3)                  
    // if begin= (0, 1), size=(1, 2)
    // then output is
    // [
    //   [2 3]
    // ]
    if (node.op() != "Slice") return false;
    
    const std::vector<int32> inputs_shape = InferenceContext::Dims(ic->input(0));
    const int input_rank = inputs_shape.size();

    const Tensor* begin = ic->input_tensor(1);
    const gtl::ArraySlice<int> begin_array(begin->flat<int>().data(), input_rank);
    VLOG(1) << node.op() << " begin " << DebugString(begin_array);

    const Tensor* size = ic->input_tensor(2);
    const gtl::ArraySlice<int> size_array(size->flat<int>().data(), input_rank);
    VLOG(1) << node.op() << " size " << DebugString(size_array);

    CHECK(diff_dims.size() > 0); // This always true, because 
                                 // it will not be here if diff_dims.size() <= 0
    const auto& input_dims = diff_dims[0];
    for (int dim: input_dims) {
      if (begin_array[dim] != 0 || size_array[dim] != inputs_shape[dim]) {
        LOG(WARNING) << "Validate " << node.name() << "(" << node.op() << ") xla auto padding rule failed; "
                     << "input padding dim=" << dim
                     << "and begin=" << begin_array[dim] << " size=" << size_array[dim];
        return false;
      }
    }
    return true;
  }

  inline bool ValidateStridedSlice(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // StridedSlice behavies like Slice
    // Slice piece along the padding dim must equal to the dim of input
    // That is, begin=0, end=input shape and strides=1

    // eg: Input shape is (d0, d1, d2 ... dn)
    // If pad dim is 0, then
    // begin must be [0, x, x, x] and end must be [d0, x, x, x],
    // slice must be [1, x, x, x]

    // How does StridedSlice works?
    // eg: input  
    // [
    //     [
    //          [1,1,1]
    //          [2,2,2]
    //     ]
    //     [
    //         [3,3,3]
    //         [4,4,4]
    //     ]
    //     [
    //         [5,5,5]
    //         [6,6,6]
    //     ]
    //]
    // so the input shape is (3, 2, 3) 
    // if begin= (0, 0, 0), end=(2, 2, 2), stride=(1,2,1)
    // then output is
    // [
    //   [
    //     [1,1],
    //     [3,3]
    //   ]
    // ]
    // mask introduction: https://www.codeleading.com/article/55641533577/
    if (node.op() != "StridedSlice") return false;
    
    const std::vector<int32> inputs_shape = InferenceContext::Dims(ic->input(0));
    const int input_rank = inputs_shape.size();

    const Tensor* begin = ic->input_tensor(1);
    const gtl::ArraySlice<int> begin_array(begin->flat<int>().data(), input_rank);
    VLOG(1) << "begin " << DebugString(begin_array);

    const Tensor* end = ic->input_tensor(2);
    const gtl::ArraySlice<int> end_array(end->flat<int>().data(), input_rank);
    VLOG(1) << "end " << DebugString(end_array);

    const Tensor* strides = ic->input_tensor(3);
    const gtl::ArraySlice<int> strides_array(strides->flat<int>().data(), input_rank);
    VLOG(1) << "strides " << DebugString(strides_array);

    int begin_mask = node.attr().at("begin_mask").i();
    int end_mask = node.attr().at("end_mask").i();
    int ellipsis_mask = node.attr().at("ellipsis_mask").i();
    int shrink_axis_mask = node.attr().at("shrink_axis_mask").i();

    //TODO add support for new_axis_mask
    int new_axis_mask = node.attr().at("new_axis_mask").i();
    //if (new_axis_mask != 0) {
    //  LOG(WARNING) << "Validate " << node.name() << "(" << node.op() << ") xla auto padding rule failed";
    //  LOG(WARNING) << ", which is supported only" 
    //               << " when new_axis_mask is 0";
    //  return false;
    //}

    CHECK(diff_dims.size() > 0); // This always true, because 
                                 // it will not be here if diff_dims.size() <= 0
    const auto& input_dims = diff_dims[0];
    for (int dim: input_dims) {
      int begin_bit_mask = (begin_mask >> dim) & 1;
      int end_bit_mask = (end_mask >> dim) & 1;
      int ellipsis_bit_mask = (ellipsis_mask >> dim) & 1;
      int shrink_bit_mask = (shrink_axis_mask >> dim) & 1;
      if ((begin_array[dim] != 0 && begin_bit_mask != 1) || 
          (end_array[dim] != inputs_shape[dim] && end_bit_mask != 1) ||
          (strides_array[dim] != 1 && ellipsis_bit_mask != 1) ||
          (shrink_bit_mask == 1)) {
        LOG(WARNING) << "Validate " << node.name() << "(" << node.op() << ") xla auto padding rule failed";
        LOG(WARNING) << node.op() << " input shape " << DebugString(inputs_shape);
        LOG(WARNING) << node.op() << " begin " << DebugString(begin_array);
        LOG(WARNING) << node.op() << " end " << DebugString(end_array);
        LOG(WARNING) << node.op() << " strides " << DebugString(strides_array);
        LOG(WARNING) << " begin_mask=" << begin_mask << " end_mask=" << end_mask;
        LOG(WARNING) << " ellipsis_mask=" << ellipsis_mask << " shrink_axis_mask=" << shrink_axis_mask;
        LOG(WARNING) << " input padding dim=" << dim;
        return false;
      }
    }
    return true;
  }
  inline bool ValidateMatMul(const NodeDef& node, 
      const std::vector<std::vector<int>>& diff_dims, 
      InferenceContext* ic) {
    // Matmul (m, k) * (k, n), output is (m, n)
    // The padding dim cannot be k
    // If left input rank is n,
    // padding dim canot be n - 1 for left operand if no transpose
    // padding dim cannot be n - 2 for left operand with transpose

    if (node.op() != "MatMul" && node.op() != "BatchMatMulV2" &&
        node.op() != "BatchMatMul") return false;

    CHECK(diff_dims.size() == 2) << node.op() << " must have 2 inputs but get " << diff_dims.size();
    bool transpose_a = node.attr().count("transpose_a") > 0 ? node.attr().at("transpose_a").b() : false;

    int left_rank = InferenceContext::Rank(ic->input(0));
    int reduce_axis = transpose_a ? left_rank - 2 : left_rank - 1;
    if (in_vector(diff_dims[0], reduce_axis)) {
      LOG(WARNING) << "Validate " << node.name() << "(" << node.op() << ") xla auto padding rule failed; "
                   << "Padding dim and reduce dim are same, dim=" << reduce_axis;
      return false;
    }
    return true;
  }

} // namespace

/// Public method below
XlaPaddingRule::XlaPaddingRule() {}

bool XlaPaddingRule::IsWhiteListOp(std::string op) {
  // Return whether a op is in whitelist
  // If in whitelist, it will not be checked
  if (white_list_ops.find(op) != white_list_ops.end()) {
    return true;
  } 
  return false;
}

bool XlaPaddingRule::IsBlackListOp(std::string op) {
  // Return whether a op is in blacklist
  // If in blacklist, it will not in xla cluster
  if (black_list_ops.find(op) != black_list_ops.end()) {
    return true;
  } 
  return false;
}

bool XlaPaddingRule::IsShapeSensitiveOp(std::string op) {
  // Return whether a op is a shape op
  // If it is a shape op, it will produce output of type int
  // which will make xla cache miss
  if (shape_ops.find(op) != shape_ops.end()) {
    return true;
  } 
  return false;
}

bool XlaPaddingRule::ValidateByOp(const NodeDef& node, 
    const std::vector<std::vector<int>>& diff_dims, 
    InferenceContext* ic) {
  // Validate whether a op is satisfy the xla padding rule

  VLOG(1) << "Validate op " << node.name() << " (" << node.op() << ") ";
  for (size_t i = 0; i < diff_dims.size(); i++) {
    std::string dims_str;
    for (auto dim: diff_dims[i]) {
      dims_str += std::to_string(dim) + " ";
    }
    std::string msg = dims_str.empty() ? " same": (" diff in dim " + dims_str);
    VLOG(1) << "input " << i << msg;
  }  
  if (diff_dims.size() <= 0) return true;

  if (ValidateMatMul(node, diff_dims, ic)) return true;
  else if (ValidateConcat(node, diff_dims, ic)) return true;
  else if (ValidateTile(node, diff_dims, ic)) return true;
  else if (ValidateReshape(node, diff_dims, ic)) return true;
  else if (ValidateSplit(node, diff_dims, ic)) return true;
  else if (ValidateSlice(node, diff_dims, ic)) return true;
  else if (ValidateStridedSlice(node, diff_dims, ic)) return true;
  else if (ValidateSoftmax(node, diff_dims, ic)) return true;
  else if (ValidateUnpack(node, diff_dims, ic)) return true;
  else {
    LOG(ERROR) << "Validate op " << node.name() << "(" << node.op() << ") failed! Cannot compile Cluster";
    return false;
  }
}

TensorShape CreateShapeFromShapeHandle(ShapeHandle s) {
  const std::vector<int32> inputs_shape = InferenceContext::Dims(s);
  TensorShape shape;
  for (int32 d: inputs_shape) {
    shape.AddDim(d);
  }
  return shape;
}

bool FindDiffDims(const TensorShape& s1, const TensorShape& s2,
    std::vector<int>& diff_dims) {
  diff_dims.clear();
  if (s1.dims() > 0 && s2.dims() > 0) {
	if (s1.dims() != s2.dims()) return false;

    for (int i = 0; i < s1.dims(); i++) {
	  if (s1.dim_size(i) != s2.dim_size(i)) {
		VLOG(1) << "Dim " << i << " " << s1.dim_size(i) << " vs " << s2.dim_size(i);
	    diff_dims.push_back(i);
	  }
	}
  }
  return true;
}

std::vector<TensorShape> XlaPaddingRule::SaveNodeInputShape(
    const NodeDef* node, InferenceContext* ic, bool is_fast_mode) {
  if (ic->num_inputs() == 0) return {};

  mutex_lock lock(cluster_validate_state_mu_);
  auto iter = node_input_shapes_.find(node->name());
  if (iter == node_input_shapes_.end() || !is_fast_mode) {
    std::vector<TensorShape> input_shapes(ic->num_inputs());
    for (int i = 0; i < ic->num_inputs(); i++) {
      if (!InferenceContext::RankAndDimKnown(ic->input(i))) {
        VLOG(1) << node->name() << " has unkown shape ";
        return {};
      }
      if (!ic->input(i).Handle()) {
        VLOG(1) << node->name() << " has unkown shape ";
        return {};
      }
      input_shapes[i] = CreateShapeFromShapeHandle(ic->input(i));
      VLOG(1) << node->name() << " save input shape " << input_shapes[i].DebugString();
    }

    node_input_shapes_[node->name()] = input_shapes;
    return {};
  } else {
    return iter->second;
  }
}

void XlaPaddingRule::CheckNodeIsPaddingValid(
    const NodeDef& node, InferenceContext* ic, 
    bool is_fast_mode) {
  if (GetGraphPaddingValid() != PaddingState::UNKNOWN) return;
  if (IsWhiteListOp(node.op())) return;

  std::vector<std::vector<int>> diff_dims_all_inputs;

  auto cached_shapes = SaveNodeInputShape(&node, ic, is_fast_mode);
  if (cached_shapes.size() == 0) return;

  for (int i = 0; i < ic->num_inputs(); i++) {
    TensorShape input_shape = CreateShapeFromShapeHandle(ic->input(i));
    std::vector<int> diff_dims;
    // Return false if inputs rank changed
    // We Only support inputs with same rank
    if (!FindDiffDims(cached_shapes[i], input_shape, diff_dims)) {
      SetGraphPaddingValid("", PaddingState::INVALID);
      return;
    }

    diff_dims_all_inputs.push_back(diff_dims);
    VLOG(1) << node.op() << " input " << i << cached_shapes[i].DebugString()
            << " vs " << input_shape.DebugString();
  }

  if (!ValidateByOp(node, diff_dims_all_inputs, ic)) {
    SetGraphPaddingValid("", PaddingState::INVALID);
  }
  VLOG(1) << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<";
}

PaddingState XlaPaddingRule::GetGraphPaddingValid(const std::string& inputs_signature) {
  mutex_lock lock(cluster_validate_state_mu_);
  if (cluster_validate_state_failed_) return PaddingState::INVALID;

  if (inputs_signature == "default") return PaddingState::UNKNOWN;

  auto iter = cluster_validate_state_.find(inputs_signature);
  if (iter == cluster_validate_state_.end()) {
    // inputs_signature does not contain "1", means all shape same
    if (inputs_signature.find("1") == string::npos) {
      cluster_validate_state_[inputs_signature] = PaddingState::VALID;
      return PaddingState::VALID;
    }

    return PaddingState::UNKNOWN;
  }
  return iter->second;
}

void XlaPaddingRule::SetGraphPaddingValid(const std::string& inputs_signature, PaddingState val) {
  mutex_lock lock(cluster_validate_state_mu_);
  cluster_validate_state_[inputs_signature] = val;
  if (val == PaddingState::INVALID) cluster_validate_state_failed_ = val;

  VLOG(1) << "SetGraphPaddingValid " << inputs_signature << " " << val;
}

void XlaPaddingRule::ForceRestPaddingValid() {
  mutex_lock lock(cluster_validate_state_mu_);
  cluster_validate_state_.clear();
  cluster_validate_state_failed_ = false;
}

}  // xla_padding_rule
}  // end namespace tensorflow

