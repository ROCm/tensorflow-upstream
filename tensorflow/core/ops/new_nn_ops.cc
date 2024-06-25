/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {
REGISTER_OP("GemmLayernormGemmSoftmaxGemm")
  .Input("matrix_a0: dataTP")        //input0 query
  .Input("matrix_b0: dataTP")        //input1 linear projection pf query
  .Input("matrix_c0: dataTP")        //input2 linaer projection of query
  .Input("layernorm_beta: dataTP")   //input3 layernorm beta of query
  .Input("layernorm_gamma: dataTP")  //input4 layernorm gamma of query
  .Input("matrix_b1: dataTP")        //input5 key after layernorm
  .Input("softmaxmask: dataTP")      //input6 keymask
  .Input("matrix_b2: dataTP")        //input7 value after linear projection
  .Output("output: dataTP")
  .Attr("dataTP: {half}")
  .Attr("head_num: int >= 1")     //head_num
  .Attr("lrelu_alpha: float")
  .Attr("do_layer_norm: bool = true")
  .Attr("do_leaky_relu: bool = false")
  .Attr("do_query_mask: bool = false")
  .SetShapeFn([](InferenceContext* c) {

    shape_inference::ShapeHandle unused_handle;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused_handle));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &unused_handle));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused_handle));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &unused_handle));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &unused_handle));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 3, &unused_handle));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &unused_handle));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 3, &unused_handle));

    shape_inference::DimensionHandle unused_dhandle;
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 2), c->Dim(c->input(1), 1), &unused_dhandle));
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 0), c->Dim(c->input(2), 0), &unused_dhandle));
    // TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(5), 1), c->Dim(c->input(6), 1), &unused_dhandle));
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(2), 0), c->Dim(c->input(5), 2), &unused_dhandle));
    // TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(6), 1), c->Dim(c->input(7), 0), &unused_dhandle));

    ShapeHandle s0 = c->input(0);
    ShapeHandle s5 = c->input(5);
    ShapeHandle s7 = c->input(7);
    ShapeHandle out;

    DimensionHandle B_kv_x_M;
    (void)c->Multiply(c->Dim(s0, 0), c->Dim(s5, 0), &B_kv_x_M);

    if (c->RankKnown(s0)) {
      std::vector<DimensionHandle> dims;
      dims.push_back(B_kv_x_M);
      dims.push_back(c->Dim(s0, 1));
      dims.push_back(c->Dim(s7, 2));
      out = c->MakeShape(dims);
    } else {
      out = c->UnknownShape();
    }
    for (int i = 0; i < c->num_outputs(); ++i) c->set_output(i, out);
    return Status::OK();
  });

REGISTER_OP("GemmRowSoftmaxGemm")
  .Input("matrix_b0: dataTP")  //(batch, head_num, seq)
  .Input("matrix_a0: dataTP")  //(new_head, head_num)
  .Input("matrix_a1: dataTP")  //(head_num, new_head)
  .Input("kmask: dataTP")  //(batch, 1, seq)
  .Output("output: dataTP")  //(batch, head_num, seq)
  .Attr("dataTP: {half}")
  .SetShapeFn([](InferenceContext* c) {

  });

REGISTER_OP("GemmLayernormGemm")
  .Input("matrix_a0: dataTP")  //input0 query
  .Input("matrix_b0: dataTP")  //input1 linear projection pf query
  .Input("matrix_c0: dataTP")  //input2 linaer projection of query
  .Input("layernorm_beta: dataTP")  //input3 layernorm beta of query
  .Input("layernorm_gamma: dataTP")  //input4 layernorm gamma of query
  .Input("matrix_b1: dataTP")  //input5 key after layernorm
  .Output("output: dataTP")
  .Attr("dataTP: {half}")
  .Attr("head_num: int >= 1")//head_num
  .Attr("lrelu_alpha: float")
  .Attr("do_layer_norm: bool = true")
  .Attr("do_leaky_relu: bool = false")
  .SetShapeFn([](InferenceContext* c) {

  });

REGISTER_OP("GemvSoftmaxGemv")
  .Input("matrix_a0: dataTP")
  .Input("matrix_b0: dataTP")
  .Input("matrix_key: dataTP")
  .Input("matrix_b1: dataTP")
  .Output("output: dataTP")
  .Attr("dataTP: {float, half}")
  .SetShapeFn([](InferenceContext* c) {

  });

REGISTER_OP("GatherGemv")
  .Input("matrix_a0: dataTP")
  .Input("matrix_b0: dataTP")
  .Input("indices: int32")
  .Output("output: dataTP")
  .Attr("dataTP: {float, half}")
  .Attr("head_num: int = 1")
  .SetShapeFn([](InferenceContext* c) {

  });

REGISTER_OP("GatherGemv2")
  .Input("matrix_a0: dataTP")
  .Input("matrix_b0: dataTP")
  .Input("indices: int32")
  .Output("output: dataTP")
  .Attr("dataTP: {float, half}")
  .Attr("head_num: int = 1")
  .SetShapeFn([](InferenceContext* c) {

  });

REGISTER_OP("GatherAttention")
  .Input("matrix_a0: dataTP")
  .Input("matrix_b0: dataTP")
  .Input("matrix_key: dataTP")
  .Input("indices: int32")
  .Input("matrix_b1: dataTP")
  .Output("output: dataTP")
  .Attr("dataTP: {float, half}")
  .Attr("head_num: int = 1")
  .SetShapeFn([](InferenceContext* c) {

  });

REGISTER_OP("FusedTileGemm")
  .Input("matrix_a0: dataTP")
  .Input("matrix_b0: dataTP")
  .Output("output: dataTP")
  .Attr("dataTP: {half}")
  .Attr("head_num: int >= 1")  //head_num
// (batch, head_num, seq) * (1, seq, head_num * head_sz)
// (batch, 1, head_num * head_sz)
  .SetShapeFn([](InferenceContext* c) {

  });

REGISTER_OP("FusedGemmBiasAdd")
  .Input("matrix_a0: dataTP")
  .Input("matrix_b0: dataTP")
  .Input("matrix_c0: dataTP")
  .Output("output: dataTP")
  .Attr("dataTP: {half}")
  // (M, K) * (N, K)
  .SetShapeFn([](InferenceContext* c) {

    shape_inference::ShapeHandle unused_handle;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused_handle));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &unused_handle));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused_handle));

    int rank = c->Rank(c->input(0));
    if (rank == 2) {
         auto M = c->Dim(c->input(0), 0);
         auto N = c->Dim(c->input(1), 0);
         c->set_output(0, c->MakeShape({M, N}));
    } else if(rank == 3) {
         auto d0 = c->Dim(c->input(0), 0);
         auto d1 = c->Dim(c->input(0), 1);
         auto N = c->Dim(c->input(1), 0);
         c->set_output(0, c->MakeShape({d0, d1, N}));
    } else {
         c->set_output(0, c->UnknownShape());
    }
    return Status::OK();
  });

}  // namespace tensorflow
