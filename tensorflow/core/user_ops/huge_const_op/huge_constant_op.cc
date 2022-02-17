#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <vector>

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;


REGISTER_OP("HugeConst")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("path: string")
    .SetShapeFn([](InferenceContext* c) {
      TensorShape shape_attr;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape_attr));
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(shape_attr, &s));
      c->set_output(0, s);
      return Status::OK();
    });
