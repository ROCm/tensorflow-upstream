/* Copyright 2021 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ContainsRegex;

class TooManyBlocksTest : public GpuPjRtCodegenTest {};

TEST_F(TooManyBlocksTest, FailsWithInvalidStatus) {
  // This test ensures that invalid (too large) launch grids are caught
  // somewhere in the pipeline. The practical relevance is low, since as of
  // 2024, the inputs or outputs have to be way too large to fit on any GPU
  // anyway.
<<<<<<< HEAD
  //
  // On ROCm, the grid splitting feature (MaybeSplitGridDimensionX) splits
  // oversized grids into multiple dimensions, so compilation succeeds instead
  // of failing. Skip this test on ROCm since the behavior is intentionally
  // different.
  if (is_built_with_rocm_) {
    GTEST_SKIP() << "ROCm handles large grids via grid dimension splitting";
  }
  const char* hlo_text = R"(
HloModule primitive_computation_mul.8

ENTRY primitive_computation_mul.8 {
  parameter.1 = s8[65536] parameter(0)
  parameter.2 = s8[65536] parameter(1)
  broadcast.3 = s8[65536,65536,65536,128,16] broadcast(parameter.1), dimensions={0}
  broadcast.4 = s8[65536,65536,65536,128,16] broadcast(parameter.2), dimensions={1}
  ROOT multiply.5 = s8[65536,65536,65536,128,16] multiply(broadcast.3, broadcast.4)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo_text));

  stream_executor::StreamExecutorAddressAllocator allocator(
      backend().default_stream_executor());
  absl::StatusOr<std::unique_ptr<Executable>> failed_executable =
      backend().compiler()->RunBackend(std::move(optimized_module),
                                       backend().default_stream_executor(),
                                       &allocator);

  EXPECT_FALSE(failed_executable.ok());
  EXPECT_THAT(
      failed_executable.status().ToString(),
      ::testing::ContainsRegex("Kernel '.*fusion.*' launch needs more blocks"));
=======
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule primitive_computation_mul.8

    ENTRY primitive_computation_mul.8 {
      p.1 = s8[65536] parameter(0)
      p.2 = s8[65536] parameter(1)
      bcast.3 = s8[65536,65536,65536,128,16] broadcast(p.1), dimensions={0}
      bcast.4 = s8[65536,65536,65536,128,16] broadcast(p.2), dimensions={1}
      ROOT multiply.5 = s8[65536,65536,65536,128,16] multiply(bcast.3, bcast.4)
    }
  )"));
  EXPECT_THAT(CompileToExecutable(std::move(hlo_module),
                                  /*run_optimization_passes=*/true),
              StatusIs(_, ContainsRegex(
                              "Kernel '.*fusion.*' launch needs more blocks")));
>>>>>>> sept15
}

}  // namespace
}  // namespace xla::gpu
