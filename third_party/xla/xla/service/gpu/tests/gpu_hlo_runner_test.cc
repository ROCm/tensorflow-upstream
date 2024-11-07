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
#include <regex>
#include "xla/error_spec.h"
#include "xla/literal_comparison.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"


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


class HloRunnerTest : public GpuCodegenTest {

protected:
  constexpr static const char *CsvSep = " , ";

  void run_internal(std::istream& ifs, std::ostream& ofs) {

    const static std::pair< std::regex, const char *> replaces[] = {
        { std::regex(R"x(\[\(0\),)x"), "[" }, // remove dynamic shapes
        { std::regex(R"x(u32\[\] get-dimension-size\()x"), "s32[] get-dimension-size(" },
        { std::regex(R"x(u32\[\] %get-dimension-size\.)x"), "s32[] %get-dimension-size."},
    };

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    auto input = buffer.str();
    for (const auto& rep : replaces) {
      input = std::regex_replace(input, rep.first, rep.second);
    }

    HloModuleConfig config = GetModuleConfigForTest();
#if 1
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(input, 
          config));

  auto ref_module = module->Clone();  
  TF_ASSERT_OK_AND_ASSIGN(auto exec, test_runner_.CreateExecutable(std::move(module), true));

  VLOG(0) << "Creating fake args..";
  TF_ASSERT_OK_AND_ASSIGN(auto fake_arguments, xla::MakeFakeArguments(ref_module.get(), 
        true, /*pseudo-random*/
        false /* use large range*/));
  auto arg_ptrs = MakePointerVector<xla::Literal>(fake_arguments);

  // TF_ASSERT_OK_AND_ASSIGN(auto truth, 
  //       ReadLiteralFromProto("/tf/xla/expected.pb"));
  // TF_ASSERT_OK_AND_ASSIGN(auto truth, 
  // ref_runner.ExecuteWithExecutable(ref_exec.get(), arg_ptrs, nullptr));
  // WriteLiteralToTempFile(truth, "expected");
  //VLOG(0) << "Got expected literal from file.. running test";

  int num_runs = 100, num_warmups = 2;
  TF_ASSERT_OK_AND_ASSIGN(auto argument_buffers,
                      test_runner_.TransferLiteralsToDevice(arg_ptrs));

  uint64_t timeNs = 0;
  for(int i = 0; i < num_runs + num_warmups; i++) {
    if(i == num_warmups) {
      VLOG(0) << "Warmup finished.. running";
      ASSERT_TRUE(backend().default_stream_executor()->SynchronizeAllActivity());
    }
    xla::ExecutionProfile profile;
    //profile.set_warmup_run_executed(true);
    TF_ASSERT_OK_AND_ASSIGN(auto result,
                      test_runner_.ExecuteWithDeviceBuffers(
                          /*executable=*/exec.get(),
                          /*arguments=*/argument_buffers,
                          /*profile=*/&profile));
    if (i >= num_warmups) timeNs += profile.compute_time_ns();
    //VLOG(0) << i << " compute time: " << profile.compute_time_ns();
  }
  double usec = (double)timeNs  / (num_runs * 1000);
  VLOG(0) << "Time elapsed: " << usec << " usec";
  ofs << usec;

#if 0
  VLOG(0) << "Performing correctness check.";
  TF_ASSERT_OK_AND_ASSIGN(
       auto test_res, test_runner_.ExecuteWithExecutable(exec.get(), arg_ptrs));

  VLOG(0) << "Running reference exec..";
  auto& ref_runner = HloTestBase::reference_runner_;
  TF_ASSERT_OK_AND_ASSIGN(
       auto ref_exec, ref_runner.CreateExecutable(std::move(ref_module), true));

  TF_ASSERT_OK_AND_ASSIGN(
       auto truth, ref_runner.ExecuteWithExecutable(ref_exec.get(), arg_ptrs));

  ErrorSpec error_spec{1e-2, 1e-3};
  //ErrorSpec error_spec(1e-5 /*abs*/, 1e-5 /*rel*/);
  ASSERT_EQ(literal_comparison::Near(/*expected=*/truth,
                                  /*actual=*/test_res,
                                  /*error=*/error_spec,
                           /*detailed_message=*/true, {}), absl::OkStatus());
#endif
 //    EXPECT_TRUE(RunAndCompare(std::move(module), 
  // //     absl::Span< xla::Literal * const>(arg_ptrs.data(), arg_ptrs.size()), error_spec));
#else
  int NumReplicas = 8, NumParts = 1;
  config.set_replica_count(NumReplicas);
  config.set_num_partitions(NumParts);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(input, config));
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

};

TEST_F(HloRunnerTest, RunSingle) {

  if (std::ifstream ifs("input.hlo"); ifs.good()) {
    std::ofstream ofs;
    return run_internal(ifs, ofs);
  }
  std::ifstream ifs("pattern.txt");
  ASSERT_TRUE(ifs.good());

  std::string line;
  std::getline(ifs, line);
  VLOG(0) << "Using file pattern: " << line;

  auto env = tsl::Env::Default();
  std::string csv("hlo_runner_results.csv");
  bool exists = env->FileExists(csv).ok();
  std::ofstream ofs(csv, std::ios_base::app);

  std::vector<std::string> matches;
  auto pattern = tsl::io::JoinPath("/", line);
  ASSERT_TRUE(env->GetMatchingPaths(pattern, &matches).ok());

  if (!exists) ofs << CsvSep; // add one column for the header

  for(size_t i = 0; i < matches.size(); i++) {
    auto s = matches[i];
    auto res = s.find_last_of('/');
    if (res != std::string::npos) s = s.substr(res + 1);
    res = s.find_first_of(".");
    if (res != std::string::npos) s = s.substr(0, res);
    if (!exists) {
      ofs << s << (i == matches.size() - 1 ? "\n" : CsvSep);
    }
  }

  ofs << "tf 2.15" << CsvSep;
  for(size_t i = 0; i < matches.size(); i++) {
    auto s = matches[i];
    std::ifstream ifs(s);
    if (!ifs.good()) {
      VLOG(0) << "Skipping file: " << s;
      ofs << CsvSep;
      continue;
    }
    VLOG(0) << i << " of " << matches.size() << ": HLO test for: " 
            << s << " ---------------------";

    run_internal(ifs, ofs);
    ofs << (i == matches.size() - 1 ? "\n" : CsvSep);
    std::flush(ofs);
  }
}

}  // namespace gpu
}  // namespace xla


