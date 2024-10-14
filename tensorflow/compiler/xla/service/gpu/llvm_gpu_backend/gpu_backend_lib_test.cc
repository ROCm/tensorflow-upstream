// File: gpu_backend_lib_amdgpu_test.cc

#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"

#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/rocm_rocdl_path.h"



namespace xla {
namespace gpu {
namespace {

class AmdgpuBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    module_ = std::make_unique<llvm::Module>("gpu_backend_lib_amdgpu_test_module", context_);
    hlo_module_config_ = HloModuleConfig();
  }

  llvm::LLVMContext context_;
  std::unique_ptr<llvm::Module> module_;
  HloModuleConfig hlo_module_config_;
};

TEST_F(AmdgpuBackendTest, CompileToHsacoBasic) {
  int gpu_version = 942;  // Assuming gfx900
  std::string rocdl_dir_path = tensorflow::RocdlRoot();
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << *(module_.get());

  // Delete the first two lines, since they usually vary even when the rest of
  // the code is the same (but verify that they are what we expect).
  if (str.size() >= 13 && str.substr(0, 13) == "; ModuleID = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }
  if (str.size() >= 18 && str.substr(0, 18) == "source_filename = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) str = str.substr(pos + 1);
  }

  auto status_or_hsaco = amdgpu::CompileToHsaco(
      module_.get(), gpu_version, hlo_module_config_, rocdl_dir_path);
  
  TF_ASSERT_OK(status_or_hsaco.status());
  EXPECT_FALSE(status_or_hsaco.ValueOrDie().empty());
  std::string hsaco_cache_dir;
  tensorflow::ReadStringFromEnvVar("TF_XLA_HSACO_CACHE_DIR", "/tmp",
                                     &hsaco_cache_dir);
  std::string hsaco_filename =
        absl::StrCat(std::hash<std::string>{}(str), ".gfx",
                     std::to_string(gpu_version), ".hsaco");
  std::string hsaco_path = tensorflow::io::JoinPath(hsaco_cache_dir, hsaco_filename);
  EXPECT_TRUE(tensorflow::Env::Default()->FileExists(hsaco_path).ok());
}

TEST_F(AmdgpuBackendTest, CompileToHsacoInvalidGpuVersion) {
  GpuVersion gpu_version = std::pair<int, int>{10, 0};  // Invalid for AMDGPU
  std::string rocdl_dir_path = tensorflow::RocdlRoot();
  
  auto status_or_hsaco = amdgpu::CompileToHsaco(
      module_.get(), gpu_version, hlo_module_config_, rocdl_dir_path);
  
  EXPECT_FALSE(status_or_hsaco.ok());
}


// Add more tests as needed...

}  // namespace
}  // namespace gpu
}  // namespace xla