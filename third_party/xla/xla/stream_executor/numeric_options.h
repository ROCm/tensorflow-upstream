/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------*/

#ifndef XLA_STREAM_EXECUTOR_NUMERIC_OPTIONS_H_
#define XLA_STREAM_EXECUTOR_NUMERIC_OPTIONS_H_

namespace stream_executor {

// Options that specify the numeric behavior of operations like matrix
// multiplications and convolutions
struct NumericOptions {
  NumericOptions(bool require_determinism, bool allow_tf32, int dynamic_range1=-1, int dynamic_range2=-1)
      : require_determinism(require_determinism), allow_tf32(allow_tf32), 
      range1(DynamicRange(dynamic_range1)), range2(DynamicRange(dynamic_range2)) {}

  NumericOptions() : require_determinism(false), allow_tf32(true) {}

  // If true, the op must be deterministic
  bool require_determinism;
  // If true, float32 inputs can be rounded to TensorFloat-32 precision
  bool allow_tf32;

  // Mirrors xla_data.proto/Precision
  enum DynamicRange {
    UNDEFINED = -1,

    DEFAULT = 0,
    HIGH = 1,
    HIGHEST = 2,
    PACKED_NIBBLE = 3,
    E4B6 = 4,
    E4B8 = 5,
    E4B10 = 6,
    E4B12 = 7,
    E4B14 = 8,
    E5 = 9
  };

  DynamicRange range1=UNDEFINED;
  DynamicRange range2=UNDEFINED;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_NUMERIC_OPTIONS_H_
