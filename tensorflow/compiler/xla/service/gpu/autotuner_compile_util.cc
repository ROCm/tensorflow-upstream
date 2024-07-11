/* Copyright 2023 The OpenXLA Authors.

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

#include "tensorflow/compiler/xla/service/gpu/autotuner_compile_util.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/autotuner_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace gpu {

/* static */ StatusOr<RedzoneBuffers> RedzoneBuffers::FromInstruction(
    const HloInstruction& instruction, const AutotuneConfig& config,
    se::Stream *stream, const DebugOptions& debug_options, 
    BuffersToCreate buffers_to_create) {

  std::vector< Shape > input_shapes;
  input_shapes.reserve(instruction.operand_count());
  for (const auto* operand : instruction.operands()) {
    input_shapes.push_back(operand->shape());
  }
  return FromShapes(std::move(input_shapes), instruction.shape(), 
          config, stream, debug_options, buffers_to_create);
}

/* static */ StatusOr<RedzoneBuffers> RedzoneBuffers::FromShapes(
    std::vector<Shape>&& input_shapes, const Shape& output_shape,
    const AutotuneConfig& config, se::Stream *stream,
    const DebugOptions& debug_options, BuffersToCreate buffers_to_create) {
  RedzoneBuffers buffers;

  TF_ASSIGN_OR_RETURN(auto rz_allocator, AutotunerUtil::CreateRedzoneAllocator(
                                             config, stream, debug_options));
  buffers.redzone_allocator_ =
      std::make_unique<se::RedzoneAllocator>(std::move(rz_allocator));

  int64 rng_state = 0;
  TF_RETURN_IF_ERROR(
      buffers.CreateInputs(std::move(input_shapes), config, rng_state));

  if (buffers_to_create == BuffersToCreate::kAllInputsAllOutputs ||
      buffers_to_create == BuffersToCreate::kAllInputsOutputsNoScratch) {
    TF_RETURN_IF_ERROR(buffers.CreateOutputs(output_shape, config, 
                                             buffers_to_create, rng_state));
  }
  return buffers;
}

Status RedzoneBuffers::CreateInputs(std::vector<Shape>&& input_shapes,
                             const AutotuneConfig& config, int64& rng_state) {
  input_shapes_ = std::move(input_shapes);
  for (const auto& shape : input_shapes_) {
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase buf,
        AutotunerUtil::CreateBuffer(*redzone_allocator_, shape,
                                    config, rng_state));
    input_buffers_.push_back(buf);
  }
  return Status::OK();
}

Status RedzoneBuffers::CreateOutputs(const Shape& output_shape,
                                           const AutotuneConfig& config,
                                           BuffersToCreate buffers_to_create,
                                           int64& rng_state) {
  if (!output_shape.IsTuple()) {
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase buf,
        AutotunerUtil::CreateBuffer(*redzone_allocator_, output_shape,
                                    config, rng_state));
    output_buffers_.push_back(buf);
    output_shape_ = output_shape;
    return Status::OK();
  }

  // The output is a tuple.

  auto current_shape_it = output_shape.tuple_shapes().begin();
  auto end = output_shape.tuple_shapes().end();
  end -= buffers_to_create == kAllInputsAllOutputs ? 0 : 1;

  output_shape_ = std::distance(current_shape_it, end) == 1
                      ? *current_shape_it
                      : ShapeUtil::MakeTupleShape(
                            std::vector<Shape>{current_shape_it, end});

  for (; current_shape_it < end; current_shape_it++) {
    if (current_shape_it->IsTuple()) {
      return Unimplemented("Nested tuples are unsupported by RedzoneBuffers.");
    }
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase buf,
        AutotunerUtil::CreateBuffer(*redzone_allocator_, *current_shape_it,
                                    config, rng_state));
    output_buffers_.push_back(buf);
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
