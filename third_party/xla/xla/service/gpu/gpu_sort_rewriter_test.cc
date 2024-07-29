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

#include "xla/service/gpu/gpu_sort_rewriter.h"

#include <utility>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class GpuSortRewriterTest : public HloTestBase {
 public:
  void SetUp() override {
    HloTestBase::SetUp();
    GpuSortRewriter::SetSortSizeThresholdForTestingOnly(1000);
  }

  bool RunModuleAndPass(HloModule* module) {
    auto cloned = module->Clone();
    bool changed = GpuSortRewriter().Run(module).value();
    if (changed) {
      // Here we run an end to end test to make sure that GpuSortRewriter does
      // not introduce an incorrect rewrite. To do this, we need to clone the
      // original module because the interpreter cannot process the already
      // optimized module.
      EXPECT_TRUE(RunAndCompare(std::move(cloned), ErrorSpec{0, 0}));
    }
    return changed;
  }

  void ExpectDirection(const HloInstruction* instruction, bool descending) {
    auto config = instruction->backend_config<xla::SortOptions>();
    EXPECT_EQ(config->descending(), descending);
  }
};

// Basic sort: ascending.
TEST_F(GpuSortRewriterTest, SortKeysLessThan) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Basic sort: descending.
TEST_F(GpuSortRewriterTest, SortKeysGreaterThan) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %gt = pred[] compare(%lhs, %rhs), direction=GT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/true);
}

// Comparer swaps the parameter order -> direction is reversed.
TEST_F(GpuSortRewriterTest, SortKeysGreaterThanSwapped) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(1)
  %rhs = f32[] parameter(0)
  ROOT %gt = pred[] compare(%lhs, %rhs), direction=GT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Sort a pair of tensors, keys go first.
TEST_F(GpuSortRewriterTest, SortPairs) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs_key = u32[] parameter(0)
  %rhs_key = u32[] parameter(1)
  %lhs_value = f32[] parameter(2)
  %rhs_value = f32[] parameter(3)
  ROOT %lt = pred[] compare(%lhs_key, %rhs_key), direction=LT
}

ENTRY %main {
  %input_keys = u32[1000] parameter(0)
  %input_values = f32[1000] parameter(1)
  ROOT %sort = (u32[1000], f32[1000]) sort(%input_keys, %input_values),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(), 0),
                                  m::GetTupleElement(m::CustomCall(), 1))));
}

// Sort a pair of tensors, keys go last.
TEST_F(GpuSortRewriterTest, SortPairsSwapped) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs_value = f32[] parameter(0)
  %rhs_value = f32[] parameter(1)
  %lhs_key = u32[] parameter(2)
  %rhs_key = u32[] parameter(3)
  ROOT %lt = pred[] compare(%lhs_key, %rhs_key), direction=LT
}

ENTRY %main {
  %input_values = f32[1000] parameter(0)
  %input_keys = u32[1000] parameter(1)
  ROOT %sort = (f32[1000], u32[1000]) sort(%input_values, %input_keys),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(), 1),
                                  m::GetTupleElement(m::CustomCall(), 0))));
}

// CUB sort doesn't support more than two tensors.
TEST_F(GpuSortRewriterTest, NoRewriteManyTensors) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  %unused1 = f64[] parameter(2)
  %unused2 = f64[] parameter(3)
  %unused3 = u64[] parameter(4)
  %unused4 = u64[] parameter(5)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input1 = f32[1000] parameter(0)
  %input2 = f64[1000] parameter(1)
  %input3 = u64[1000] parameter(2)
  ROOT %sort = (f32[1000], f64[1000], u64[1000]) sort(%input1, %input2, %input3),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Only 1D shapes are supported.
TEST_F(GpuSortRewriterTest, NoRewriteNonMinorSortDimension) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1000,4] parameter(0)
  ROOT %sort = f32[1000,4] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Kernels are compiled for a subset of types.
TEST_F(GpuSortRewriterTest, NoRewriteUnsupportedType) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = pred[] parameter(0)
  %rhs = pred[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = pred[1000] parameter(0)
  ROOT %sort = pred[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Comparer must be a simple function.
TEST_F(GpuSortRewriterTest, NoRewriteComplexComparer) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %lhs_scaled = f32[] multiply(%lhs, f32[] constant(2))
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs_scaled, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Comparer must use adjacent input values.
TEST_F(GpuSortRewriterTest, NoRewriteMixedKeysValues) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs_key = u32[] parameter(0)
  %rhs_key = u32[] parameter(1)
  %lhs_value = u32[] parameter(2)
  %rhs_value = u32[] parameter(3)
  ROOT %mixed = pred[] compare(%rhs_key, %lhs_value), direction=LT
}

ENTRY %main {
  %input_keys = u32[1000] parameter(0)
  %input_values = u32[1000] parameter(1)
  ROOT %sort = (u32[1000], u32[1000]) sort(%input_keys, %input_values),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Small shapes do not see improvement from CUB sort.
TEST_F(GpuSortRewriterTest, NoRewriteSmallSize) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[100] parameter(0)
  ROOT %sort = f32[100] sort(%input), dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_FALSE(RunModuleAndPass(module.get()));
}

// Basic sort: with batch dimension.
TEST_F(GpuSortRewriterTest, SortWithBatchDim) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[10,100] parameter(0)
  ROOT %sort = f32[10,100] sort(%input), dimensions={1}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Basic sort: with multiple batch dimensions.
TEST_F(GpuSortRewriterTest, SortWithMultipleBatchDims) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[10,10,10] parameter(0)
  ROOT %sort = f32[10,10,10] sort(%input), dimensions={2}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::GetTupleElement(
          m::CustomCall({kCubDeviceRadixSortTarget}, m::Parameter()), 0)));
  ExpectDirection(module->entry_computation()->root_instruction()->operand(0),
                  /*descending=*/false);
}

// Sort a pair of tensors (values, indices generated by iota) with a complex
// compare.
TEST_F(GpuSortRewriterTest, SortPairsIotaComparerSimple) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = u16[] parameter(0)
  %rhs = u16[] parameter(1)
  %lhs_index = s32[] parameter(2)
  %rhs_index = s32[] parameter(3)

  cmp_indices = pred[] compare(%lhs_index, %rhs_index), direction=LT
  cmp_lr = pred[] compare(%lhs, %rhs), direction=GT
  cmp_eq = pred[] compare(%lhs, %rhs), direction=EQ

  ROOT %lt = pred[] select(cmp_eq, cmp_indices, cmp_lr)
}

ENTRY %main {
  %inputs = u16[1000] parameter(0)
  %iota = s32[1000] iota(), iota_dimension=0
  ROOT %sort = (u16[1000], s32[1000]) sort(%inputs, %iota),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(), 0),
                                  m::GetTupleElement(m::CustomCall(), 1))));
}

// Sort a pair of tensors (values, indices generated by iota) with a complex
// compare computation that matches the output of the StableSortExpander pass.
TEST_F(GpuSortRewriterTest, SortPairsIotaComparerLikeStableSortExpander) {
  constexpr char kHlo[] = R"(
HloModule TestModule

%compare {
  %lhs = u16[] parameter(0)
  %rhs = u16[] parameter(1)
  %lhs_index = s32[] parameter(2)
  %rhs_index = s32[] parameter(3)

  cmp_indices = pred[] compare(%lhs_index, %rhs_index), direction=LT
  cmp_lr = pred[] compare(%lhs, %rhs), direction=GT
  cmp_rl = pred[] compare(%rhs, %lhs), direction=GT
  cmp_eq = pred[] compare(cmp_lr, cmp_rl), direction=EQ

  ROOT %lt = pred[] select(cmp_eq, cmp_indices, cmp_lr)
}

ENTRY %main {
  %inputs = u16[1000] parameter(0)
  %iota = s32[1000] iota(), iota_dimension=0
  ROOT %sort = (u16[1000], s32[1000]) sort(%inputs, %iota),
      dimensions={0}, to_apply=%compare
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunModuleAndPass(module.get()));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(), 0),
                                  m::GetTupleElement(m::CustomCall(), 1))));
}

TEST_F(GpuSortRewriterTest, SortSizeThresholdIsSet) {
  EXPECT_EQ(GpuSortRewriter::SortSizeThreshold(), 1000);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
