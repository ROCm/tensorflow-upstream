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

#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>
#include <memory>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

// #define HIP_CHECK(call)                                            \
//   do {                                                             \
//     hipError_t err = call;                                         \
//     if (err != hipSuccess) {                                       \
//       std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__; \
//       std::cerr << " - " << hipGetErrorString(err) << std::endl;   \
//       exit(1);                                                     \
//     }                                                              \
//   } while (0)

namespace xla {
namespace {

class XlaAsanTest : public gpu::GpuCodegenTest {
 public:
  void CallMultiply() {
    HloComputation::Builder builder(TestName());

    Shape param_shape = ShapeUtil::MakeShapeWithLayout(
        F32, /*dimensions=*/{1024, 1}, /*minor_to_major=*/{1, 0});
    HloInstruction* x = builder.AddInstruction(HloInstruction::CreateParameter(
        /* parameter_number=*/0, param_shape, "x"));
    HloInstruction* y = builder.AddInstruction(HloInstruction::CreateParameter(
        /* parameter_number=*/1, param_shape, "y"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(param_shape, HloOpcode::kMultiply, x, y));

    auto hlo_module = CreateNewUnverifiedModuleWithFTZ(false);
    hlo_module->AddEntryComputation(builder.Build());

    auto gpu = PlatformUtil::GetPlatform("gpu").ValueOrDie();
    HloRunner runner(gpu);

    std::unique_ptr<Executable> executable = std::move(
        runner.CreateExecutable(std::move(hlo_module), true).ValueOrDie());

    std::vector<ScopedShapedBuffer> args;

    Shape fake_shape = ShapeUtil::MakeShapeWithLayout(
        F32, /*dimensions=*/{1, 1}, /*minor_to_major=*/{1, 0});

    for (int i = 0; i < 2; i++) {
      args.emplace_back(runner.backend()
                            .transfer_manager()
                            ->AllocateScopedShapedBuffer(
                                fake_shape, runner.backend().memory_allocator(),
                                runner.backend().default_device_ordinal())
                            .ValueOrDie());
    }

    runner
        .ExecuteWithDeviceBuffers(executable.get(), args,
                                  /*profile=*/nullptr)
        .ValueOrDie();

    // Beacuse of yhe raw hip apis bellow the test at best ensures that
    // the XLA builds code with requierd instrumentation. Not that the other
    // parts of the XLA runtime play along.

    // HIP_CHECK(hipSetDevice(0));

    // float *d_A, *d_B, *d_C;
    // HIP_CHECK(hipMalloc(&d_A, 1 * sizeof(float)));
    // HIP_CHECK(hipMalloc(&d_B, 1 * sizeof(float)));
    // HIP_CHECK(hipMalloc(&d_C, 1 * sizeof(float)));

    // // Load module from compiled file
    // hipModule_t module;
    // hipFunction_t kernel;
    // HIP_CHECK(hipModuleLoadData(&module, exec->binary().data()));
    // HIP_CHECK(hipModuleGetFunction(&kernel, module, "multiply"));

    // void* args[] = {d_C, d_A, d_B};
    // size_t size = sizeof(args);
    // void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
    //                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
    //                   HIP_LAUNCH_PARAM_END};

    // HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, 10, 1, 1, 0, nullptr,
    //                                 nullptr, config));

    // (void)(hipDeviceSynchronize());

    // HIP_CHECK(hipFree(d_A));
    // HIP_CHECK(hipFree(d_B));
    // HIP_CHECK(hipFree(d_C));
    // HIP_CHECK(hipModuleUnload(module));
  }
};

TEST_F(XlaAsanTest, Simple) {
  CallMultiply();
  // EXPECT_EXIT(CallKernel(), ::testing::ExitedWithCode(1),
  //             "AddressSanitizer: heap-buffer-overflow on amdgpu device");
}

}  // namespace
}  // namespace xla
