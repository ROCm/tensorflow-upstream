/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/costs/graph_properties.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/inputs/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

#include "tensorflow/cc/ops/function_ops.h"

namespace tensorflow {
namespace grappler {
namespace {

const char kTestDataPath[] = "core/grappler/costs/graph_properties_testdata";

REGISTER_OP("TestOpWithNoInferenceFn")
    .Input("x: float")
    .Output("y: float")
    .Doc(R"doc(
Test op with no Inference Function registered.
x: input
y: output
)doc");

class XlaPaddingRuleTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Provision a single machine with 3 cpu cores
    cluster_.reset(new SingleMachine(5 * 60, 3, 0));
    TF_CHECK_OK(cluster_->Provision());

    // This function is simply
    // out = Fill(shape, value), but
    // Fill requires values in the shape input, not just shape of it, to infer
    // output shape.
    auto f = FunctionDefHelper::Create(
        // Name
        "MyFillFunc",
        // Inputs
        {"shape: int32", "value: float"},
        // Outputs
        {"out: float"},
        // Attrs
        {},
        // Nodes
        {
            {{"a"},
             "Fill",
             {"shape", "value"},
             {{"T", DataType::DT_FLOAT}, {"index_type", DataType::DT_INT32}}},
        },
        // Returns
        {{"out", "a:output:0"}});
    function_lib_.add_function()->Swap(&f);
  }

  void TearDown() override {
    TF_CHECK_OK(cluster_->Shutdown());
    cluster_.reset();
  }

 protected:
  // Returns a string form of <p>, suitable for comparing type and shape.
  // Example output for 4-d float tensor: "float: [10,2,30,4]"
  string PropToString(const OpInfo::TensorProperties& p) {
    string s = strings::StrCat(DataTypeString(p.dtype()), ": ");
    if (p.shape().unknown_rank()) {
      strings::StrAppend(&s, "?");
    } else {
      strings::StrAppend(&s, "[");
      for (int i = 0; i < p.shape().dim_size(); ++i) {
        strings::StrAppend(&s, i == 0 ? "" : ",",
                           std::max<int64>(p.shape().dim(i).size(), -1));
      }
      strings::StrAppend(&s, "]");
    }
    return s;
  }

  // Compare values of integer (DT_INT32 or DT_INT64) tensor against expected
  // ones.
  void ExpectTensorValues(const std::vector<int64>& expected,
                          const TensorProto& tensor_proto_to_compare) {
    Tensor tensor;
    EXPECT_TRUE(tensor.FromProto(tensor_proto_to_compare));
    EXPECT_EQ(expected.size(), tensor.NumElements());
    // We're interested in only integer tensors as only shapes are exported as
    // graph properties values.
    CHECK(tensor.dtype() == DT_INT32 || tensor.dtype() == DT_INT64);
    if (tensor.dtype() == DT_INT32) {
      for (int i = 0; i < tensor.NumElements(); i++) {
        EXPECT_EQ(expected[i], tensor.flat<int32>()(i));
      }
    } else {
      for (int i = 0; i < tensor.NumElements(); i++) {
        EXPECT_EQ(expected[i], tensor.flat<int64>()(i));
      }
    }
  }

  Tensor CreateConstTensor(const gtl::ArraySlice<int> values) {
    Tensor shape_tensor(DT_INT32, TensorShape({values.size()}));
    test::FillValues<int>(&shape_tensor, values);
    return shape_tensor;
  }

  std::unique_ptr<SingleMachine> cluster_;
  FunctionDefLibrary function_lib_;
};

TEST_F(XlaPaddingRuleTest, MatmulOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  auto in1 = ops::_Arg(s.WithOpName("in1"), DT_FLOAT, 0);

  auto output = ops::MatMul(s.WithOpName("output"), in0, in1);
  auto y = ops::_Retval(s.WithOpName("y"),
                           output, 0);

  std::vector<string> input_names = {"in0", "in1"};
  std::vector<string> fetch = {"y:0"};

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({128, 64})), 
      Tensor(DT_FLOAT, TensorShape({64, 96}))};
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;
  properties.ForceRestPaddingState();

  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // For matmul, (m, k) * (k, n)
  // input shape diff in m, n is Valid
  // input shape diff in k is invalid 
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({120, 64})), 
      Tensor(DT_FLOAT, TensorShape({64, 96}))};
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  std::vector<Tensor> input_tensors2 = {
      Tensor(DT_FLOAT, TensorShape({128, 64})), 
      Tensor(DT_FLOAT, TensorShape({64, 90}))};
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  std::vector<Tensor> input_tensors3 = {
      Tensor(DT_FLOAT, TensorShape({128, 60})), 
      Tensor(DT_FLOAT, TensorShape({60, 96}))};
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors3,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);
}

TEST_F(XlaPaddingRuleTest, BatchMatMulV2) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  auto in1 = ops::_Arg(s.WithOpName("in1"), DT_FLOAT, 0);

  auto output = ops::BatchMatMulV2(s.WithOpName("output"), in0, in1);
  auto y = ops::_Retval(s.WithOpName("y"),
                           output, 0);

  std::vector<string> input_names = {"in0", "in1"};
  std::vector<string> fetch = {"y:0"};

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({2, 128, 64})), 
      Tensor(DT_FLOAT, TensorShape({64, 96}))};
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;
  properties.ForceRestPaddingState();

  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // For matmul, (m, k) * (k, n)
  // input shape diff in m, n is Valid
  // input shape diff in k is invalid 
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({2, 120, 64})), 
      Tensor(DT_FLOAT, TensorShape({64, 96}))};
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  std::vector<Tensor> input_tensors2 = {
      Tensor(DT_FLOAT, TensorShape({2, 128, 64})), 
      Tensor(DT_FLOAT, TensorShape({64, 90}))};
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  std::vector<Tensor> input_tensors3 = {
      Tensor(DT_FLOAT, TensorShape({2, 128, 60})), 
      Tensor(DT_FLOAT, TensorShape({60, 96}))};
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors3,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);
}
TEST_F(XlaPaddingRuleTest, SoftmaxOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto logits = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  auto softmax = ops::Softmax(s.WithOpName("y"), logits);

  std::vector<string> input_names = {"in0"};
  std::vector<string> fetch = {"y:0"};

  // Tensor input(DT_FLOAT, TensorShape({batch_size, node_depth}));
  // input.flat<float>().setRandom();

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({128, 64}))}; 
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;

  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // axis = -1. As dim_size = 2, so axis=1
  // dynamic dim = 0 ok 
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({120, 64}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  // axis = -1. As dim_size = 2, so axis=1
  // dynamic dim = 1 no 
  std::vector<Tensor> input_tensors2 = {
      Tensor(DT_FLOAT, TensorShape({128, 60}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);
}

TEST_F(XlaPaddingRuleTest, ConcatOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  auto in1 = ops::_Arg(s.WithOpName("in1"), DT_FLOAT, 0);
  auto in2 = ops::_Arg(s.WithOpName("in2"), DT_FLOAT, 0);
  std::vector<Output> inputs{
      in0,
      in1,
      in2,
  }; 

  Output axis = ops::Const(s.WithOpName("axis"), 0, {});
  Output concat0 = ops::Concat(s.WithOpName("y"), inputs, axis);

  std::vector<string> input_names = {"in0", "in1", "in2"};
  std::vector<string> fetch = {"y:0"};

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({128, 64})), 
      Tensor(DT_FLOAT, TensorShape({64, 64})), 
      Tensor(DT_FLOAT, TensorShape({96, 64}))};
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;
  properties.ForceRestPaddingState();

  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;

  // axis = 0
  // dynamic dim = 1 ok 
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({128, 60})), 
      Tensor(DT_FLOAT, TensorShape({64, 60})), 
      Tensor(DT_FLOAT, TensorShape({96, 60}))};
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  // axis = 0
  // dynamic dim = 0 no 
  std::vector<Tensor> input_tensors2 = {
      Tensor(DT_FLOAT, TensorShape({120, 64})), 
      Tensor(DT_FLOAT, TensorShape({60, 64})), 
      Tensor(DT_FLOAT, TensorShape({90, 64}))};
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);
}

TEST_F(XlaPaddingRuleTest, SplitOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  auto axis = ops::Const(s.WithOpName("axis"), 1, {});
  auto split = ops::Split(s.WithOpName("y"), axis, in0, 4);
  std::vector<string> input_names = {"in0"};
  std::vector<string> fetch = {"y:0"};

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({128, 64}))}; 
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;

  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // axis = 1
  // dynamic dim = 0 ok 
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({120, 64}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  // axis = 1
  // dynamic dim = 1 no 
  std::vector<Tensor> input_tensors2 = {
      Tensor(DT_FLOAT, TensorShape({128, 60}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);
}

TEST_F(XlaPaddingRuleTest, SliceOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  auto begin = ops::Const(s.WithOpName("begin"), {0, 1});
  auto size = ops::Const(s.WithOpName("size"), {120, 10});
  auto split = ops::Slice(s.WithOpName("y"), in0, begin, size);
  std::vector<string> input_names = {"in0"};
  std::vector<string> fetch = {"y:0"};

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({128, 64}))}; 
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;

  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // axis = 1
  // dynamic dim = 0 ok 
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({120, 64}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  // axis = 1
  // dynamic dim = 1 no 
  std::vector<Tensor> input_tensors2 = {
      Tensor(DT_FLOAT, TensorShape({128, 70}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);

}

TEST_F(XlaPaddingRuleTest, StridedSliceOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  ops::StridedSlice::Attrs attrs = ops::StridedSlice::Attrs()
                                       .BeginMask(2)
                                       .EndMask(1)
                                       .EllipsisMask(2)
                                       .NewAxisMask(0)
                                       .ShrinkAxisMask(0);

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  // begin_mask=0b10, actually is {0, 0}
  auto begin = ops::Const(s.WithOpName("begin"), {0, 1});
  // end_mask=0b01, actually is {0, 10}
  auto end = ops::Const(s.WithOpName("end"), {100, 10});
  // ellipsis_mask=0b10, actually is {1, 1}
  auto strides = ops::Const(s.WithOpName("strides"), {1, 2});
  auto out = ops::StridedSlice(s.WithOpName("y"), in0, begin, end, strides, attrs);

  std::vector<string> input_names = {"in0"};
  std::vector<string> fetch = {"y:0"};

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({128, 64}))}; 
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;

  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // dynamic dim = 1 
  // begin=0, end=10, strides=1, no
  std::vector<Tensor> input_tensors2 = {
      Tensor(DT_FLOAT, TensorShape({128, 70}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);

  // dynamic dim = 0 
  // begin=0, end=ignore, strides=1 ok
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({120, 64}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);
}

TEST_F(XlaPaddingRuleTest, StridedSliceOp1) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  ops::StridedSlice::Attrs attrs = ops::StridedSlice::Attrs()
                                       .BeginMask(2)
                                       .EndMask(1)
                                       .EllipsisMask(2)
                                       .NewAxisMask(0)
                                       .ShrinkAxisMask(1);

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  // begin_mask=0b10, actually is {0, 0}
  auto begin = ops::Const(s.WithOpName("begin"), {0, 1});
  // end_mask=0b01, actually is {0, 10}
  auto end = ops::Const(s.WithOpName("end"), {100, 10});
  // ellipsis_mask=0b10, actually is {1, 1}
  auto strides = ops::Const(s.WithOpName("strides"), {1, 2});
  auto out = ops::StridedSlice(s.WithOpName("y"), in0, begin, end, strides, attrs);

  std::vector<string> input_names = {"in0"};
  std::vector<string> fetch = {"y:0"};

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({128, 64}))}; 
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;

  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // dynamic dim = 0 
  // begin=0, end=ignore, strides=1, 
  // shtrink=1 no (shtrink not equal to dynamic dim) 
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({120, 64}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);
}

TEST_F(XlaPaddingRuleTest, TileOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  auto multiplies = ops::Const(s.WithOpName("multiplies"), {1, 8});
  auto tile = ops::Tile(s.WithOpName("y"), in0, multiplies);
  std::vector<string> input_names = {"in0"};
  std::vector<string> fetch = {"y:0"};

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({128, 64}))}; 
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;

  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // dynamic dim = 0 
  // Tile in dim 0 is 1, ok
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({120, 64}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  // dynamic dim = 1 
  // Tile in dim 1 is 8, no
  std::vector<Tensor> input_tensors2 = {
      Tensor(DT_FLOAT, TensorShape({128, 70}))}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);

}

TEST_F(XlaPaddingRuleTest, ReshapeOp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  auto shape = ops::_Arg(s.WithOpName("shape"), DT_INT32, 0);;
  auto reshape = ops::Reshape(s.WithOpName("y"), in0, shape);
  std::vector<string> input_names = {"in0", "shape"};
  std::vector<string> fetch = {"y:0"};

  std::vector<Tensor> input_tensors = {
      Tensor(DT_FLOAT, TensorShape({32, 3, 2, 2, 5})),
      CreateConstTensor({16, 2, 3, -1, 10})}; 

  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;

  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // dynamic dim = 2 
  std::vector<Tensor> input_tensors1 = {
      Tensor(DT_FLOAT, TensorShape({32, 3, 3, 2, 5})),
      CreateConstTensor({16, 2, 3, -1, 10})}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  // dynamic dim = 1 
  // Tile in dim 1 is 8, no
  std::vector<Tensor> input_tensors2 = {
      Tensor(DT_FLOAT, TensorShape({32, 3, 4, 2, 5})), 
      CreateConstTensor({16, 1, 3, -1, 20})}; 
  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);
}

TEST_F(XlaPaddingRuleTest, MultiInputChange) {
  // in0 -> Tile    -->Add
  // in1 -> Reshape /

  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto in0 = ops::_Arg(s.WithOpName("in0"), DT_FLOAT, 0);
  auto in1 = ops::_Arg(s.WithOpName("in1"), DT_FLOAT, 0);
  auto multiplies = ops::Const(s.WithOpName("multiplies"), {2, 1});
  auto tile = ops::Tile(s.WithOpName("tile"), in0, multiplies);
  auto shape = ops::Const(s.WithOpName("multiplies"), {128, -1});
  auto reshape = ops::Reshape(s.WithOpName("reshape"), in1, shape);
  auto add = ops::Add(s.WithOpName("y"), tile, reshape);
  std::vector<string> input_names = {"in0", "in1"};
  std::vector<string> fetch = {"y:0"};

  // in0 (64, 128) ->Tile(2, 1) -> output(128, 128)
  // in1 (64, 2, 64) ->Reshape(128, -1) -> output(128, 128)
  std::vector<Tensor> input_tensors = {
    Tensor(DT_FLOAT, TensorShape({64, 128})),
    Tensor(DT_FLOAT, TensorShape({64, 2, 64})),
  }; 
  std::vector<std::pair<string, Tensor>> feed;
  for(int i = 0; i < input_names.size(); i++) {
    feed.push_back({input_names[i], input_tensors[i]});
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphProperties properties(item);
  item.fetch = fetch;
  item.feed = feed;

  properties.ForceRestPaddingState();
  TF_CHECK_OK(properties.InferStatically(
                                false, /*assume_valid_feeds*/
                                false, /*aggressive_shape_inference*/
                                false, /*include_input_tensor_values*/
                                false, /*include_output_tensor_values*/
                                input_tensors,
                                2));

  std::vector<TensorShapeProto> inferred_shape_protos;
 
  // case 1
  // in0 (64, 64) ->Tile(2, 1) -> output(64, 128)           // pass
  // in1 (64, 2, 64) ->Reshape(128, -1) -> output(128, 32)  // no change, pass
  std::vector<Tensor> input_tensors1 = {
    Tensor(DT_FLOAT, TensorShape({64, 64})),
    Tensor(DT_FLOAT, TensorShape({64, 2, 64})),
  }; 
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors1,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() == 0);

  // case 2
  // in0 (64, 128) ->Tile(2, 1) -> output(64, 128)          // pass
  // in1 (64, 1, 64) ->Reshape(128, -1) -> output(128, 32)  // fail
  std::vector<Tensor> input_tensors2 = {
    Tensor(DT_FLOAT, TensorShape({64, 128})),
    Tensor(DT_FLOAT, TensorShape({64, 1, 64})),
  }; 
  TF_CHECK_OK(properties.InferStaticallyFastMode(
              input_tensors2,
              inferred_shape_protos));
  EXPECT_TRUE(properties.GetXlaPaddingState() < 0);

}


}  // namespace
}  // namespace grappler
}  // namespace tensorflow
