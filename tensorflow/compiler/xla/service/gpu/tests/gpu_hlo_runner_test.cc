/* Copyright 2022 The OpenXLA Authors.

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

#include <fstream>
#include <sstream>
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace xla {
namespace gpu {

template <class T>
std::vector<T*> MakePointerVector(std::vector<T>& input_vec) {
  std::vector<T*> output_pointers;
  output_pointers.reserve(input_vec.size());
  for (auto& input : input_vec) {
    output_pointers.push_back(&input);
  }
  return output_pointers;
}


class HloRunnerTest : public GpuCodegenTest {};

TEST_F(HloRunnerTest, RunSingle) {

  std::ifstream ifs("input.hlo");
  ASSERT_TRUE(ifs.good());

  std::stringstream buffer;
  buffer << ifs.rdbuf();

  HloModuleConfig config = GetModuleConfigForTest();
#if 1
  //config.set_num_partitions(8);

TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(buffer.str(), 
          config));
  ErrorSpec error_spec{1e-2, 1e-3};
  auto ref_module = module->Clone();  
  TF_ASSERT_OK_AND_ASSIGN(auto exec, test_runner_.CreateExecutable(std::move(module), true));

  VLOG(0) << "Creating fake args..";
  auto fake_arguments = xla::MakeFakeArguments(ref_module.get(), 
        false, /*pseudo-random*/
        false /* use large range*/).ValueOrDie();
  auto arg_ptrs = MakePointerVector<xla::Literal>(fake_arguments);

  // auto& ref_runner = HloTestBase::reference_runner_;
  // TF_ASSERT_OK_AND_ASSIGN(
  //     auto ref_exec, ref_runner.CreateExecutable(std::move(ref_module), true));

  // TF_ASSERT_OK_AND_ASSIGN(auto truth, 
  //       ReadLiteralFromProto("/tf/xla/expected.pb"));
  // TF_ASSERT_OK_AND_ASSIGN(auto truth, 
  // ref_runner.ExecuteWithExecutable(ref_exec.get(), arg_ptrs, nullptr));
  // WriteLiteralToTempFile(truth, "expected");
  //VLOG(0) << "Got expected literal from file.. running test";

    TF_ASSERT_OK_AND_ASSIGN(auto test_res, 
         test_runner_.Execute(std::move(exec), arg_ptrs, nullptr));

  // EXPECT_TRUE(RunAndCompare(std::move(module), 
  //     absl::Span< xla::Literal * const>(arg_ptrs.data(), arg_ptrs.size()), error_spec));
#else
  int NumReplicas = 8, NumParts = 1;
  config.set_replica_count(NumReplicas);
  config.set_num_partitions(NumParts);

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(buffer.str(), config));
  DeviceAssignment assn(/*replica_count=*/NumReplicas,
                        /*computation_count=*/NumParts);
  for (int64_t i = 0, k = 0; i < NumReplicas; i++)
  for (int64_t j = 0; j < NumParts; j++) {
    assn(i, j) = k++;
  }

  auto fake_arguments = xla::MakeFakeArguments(
      module.get(),
      true, /*pseudo-random*/
      false /* use large range*/).ValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(auto exec, 
      test_runner_.CreateExecutable(std::move(module), true));

 for(int i = 0; i < 10; i++) {
   VLOG(0) << "Running iteration #" << i;
   TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
         HloTestBase::ExecuteReplicated(
          [&](int64_t){ return exec.get(); },
          [&fake_arguments](int64_t replica_id)
          { return fake_arguments.size(); },
          [&fake_arguments](int64_t replica_id, int64_t idx)
          { return &fake_arguments[idx]; },
          NumReplicas, false /*run hlo*/, &assn));
   ASSERT_EQ(results.size(), NumReplicas);
 }
#endif
}

}  // namespace gpu
}  // namespace xla
