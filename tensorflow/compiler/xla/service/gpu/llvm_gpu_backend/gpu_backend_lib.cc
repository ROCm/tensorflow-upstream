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

#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"

#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.inc"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/dump_ir_pass.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/env_var.h"


namespace xla {
namespace gpu {
namespace {

// Inline threshold value to use in LLVM AMDGPU backend.
const int kAMDGPUInlineThreshold = 0x100000;

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

// Gets the GPU name as it's known to LLVM for a given compute
// capability.  If we see an unrecognized compute capability, we
// return the highest one that is known and below the selected device.
static std::string GetSmName(std::pair<int, int> compute_capability) {
  int compute_capability_version =
      compute_capability.first * 10 + compute_capability.second;
  int sm_version = 30;
  // If the current compute capability isn't known, fallback to the
  // most recent version before it.
  int supported_versions[] = {75, 72, 70, 62, 61, 60, 53,
                              52, 50, 37, 35, 32, 30};
  for (int v : supported_versions) {
    if (v <= compute_capability_version) {
      sm_version = v;
      break;
    }
  }

  // If the current CC isn't supported by LLVM and it is newer then
  // the max supported LLVM version, do not warn about it. The end
  // user can't do anything about this. PTX compiled for SM75 will
  // run on SM80 too.
  if (sm_version != compute_capability_version &&
      compute_capability_version < supported_versions[0]) {
    LOG(WARNING) << "Unknown compute capability (" << compute_capability.first
                 << ", " << compute_capability.second << ") ."
                 << "Defaulting to telling LLVM that we're compiling for sm_"
                 << sm_version;
  }
  return absl::StrCat("sm_", sm_version);
}

// Convenience function for producing a name of a temporary compilation product
// from the input filename.
string MakeNameForTempProduct(absl::string_view input_filename,
                              absl::string_view extension) {
  return ReplaceFilenameExtension(tensorflow::io::Basename(input_filename),
                                  extension);
}

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(llvm::PassRegistry* pass_registry) {
  llvm::initializeCore(*pass_registry);
  llvm::initializeCodeGen(*pass_registry);
  llvm::initializeScalarOpts(*pass_registry);
  llvm::initializeObjCARCOpts(*pass_registry);
  llvm::initializeVectorization(*pass_registry);
  llvm::initializeIPO(*pass_registry);
  llvm::initializeAnalysis(*pass_registry);
  llvm::initializeTransformUtils(*pass_registry);
  llvm::initializeInstCombine(*pass_registry);
  llvm::initializeInstrumentation(*pass_registry);
  llvm::initializeTarget(*pass_registry);
  llvm::initializeCodeGenPreparePass(*pass_registry);
}

// returns the targetmachine, given a triple.
std::unique_ptr<llvm::TargetMachine> GetTargetMachine(
    llvm::Triple triple, absl::string_view cpu_name,
    const HloModuleConfig& hlo_module_config, absl::string_view feature_str) {
  std::string error;
  const llvm::Target* target = TargetRegistry::lookupTarget("", triple, error);
  if (target == nullptr) {
    LOG(FATAL) << "unable to find target for triple '" << triple.str() << "'"
               << " -- " << error;
    return nullptr;
  }

  TargetOptions target_options = InitTargetOptionsFromCodeGenFlags();

  // enable fma synthesis.
  target_options.AllowFPOpFusion = FPOpFusion::Fast;

  // set the verbose assembly options.
  target_options.MCOptions.AsmVerbose = false;

  // the selection of codegen optimization level is copied from function
  // getcodegenoptlevel in //external/llvm/tools/opt/opt.cpp.
  CodeGenOpt::Level codegen_opt_level;
  switch (hlo_module_config.debug_options().xla_backend_optimization_level()) {
    case 1:
      codegen_opt_level = CodeGenOpt::Less;
      break;
    case 2:
      codegen_opt_level = CodeGenOpt::Default;
      break;
    case 3:
      codegen_opt_level = CodeGenOpt::Aggressive;
      break;
    default:
      codegen_opt_level = CodeGenOpt::None;
  }
  return absl::WrapUnique(target->createTargetMachine(
      triple.str(), llvm_ir::AsStringRef(cpu_name),
      llvm_ir::AsStringRef(feature_str), target_options, getRelocModel(),
      getCodeModel(), codegen_opt_level));
}

// Adds the standard LLVM optimization passes, based on the speed optimization
// level (opt_level) and size optimization level (size_level). Both module
// and function-level passes are added, so two pass managers are passed in and
// modified by this function.
void AddOptimizationPasses(unsigned opt_level, unsigned size_level,
                           llvm::TargetMachine* target_machine,
                           llvm::legacy::PassManagerBase* module_passes,
                           llvm::legacy::FunctionPassManager* function_passes,
                           int inline_threshold) {
  PassManagerBuilder builder;
  builder.OptLevel = opt_level;
  builder.SizeLevel = size_level;

  if (opt_level > 1) {
    builder.Inliner = llvm::createFunctionInliningPass(inline_threshold);
  } else {
    // Only inline functions marked with "alwaysinline".
    builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
  }

  builder.DisableUnrollLoops = opt_level == 0;
  builder.LoopVectorize = opt_level > 0;
  builder.SLPVectorize = opt_level > 1 && size_level < 2;

  // NVPTX's early-as-possible passes include NVVM reflect.
  target_machine->adjustPassManager(builder);

  builder.populateFunctionPassManager(*function_passes);
  builder.populateModulePassManager(*module_passes);
}

// Emits the given module to a bit code file.
void EmitBitcodeToFile(const Module& module, absl::string_view filename) {
  std::error_code error_code;
  llvm::ToolOutputFile outfile(string(filename).c_str(), error_code,
                               llvm::sys::fs::F_None);
  if (error_code) {
    LOG(FATAL) << "opening bitcode file for writing: " << error_code.message();
  }

  llvm::WriteBitcodeToFile(module, outfile.os());
  outfile.keep();
}

// Emits the given module to PTX. target_machine is an initialized TargetMachine
// for the NVPTX target.
string EmitModuleToPTX(Module* module, llvm::TargetMachine* target_machine) {
  std::string ptx;  // need a std::string instead of a ::string.
  {
    llvm::raw_string_ostream stream(ptx);
    llvm::buffer_ostream pstream(stream);
    // The extension is stripped by IrDumpingPassManager, so we need to
    // get creative to add a suffix.
    IrDumpingPassManager codegen_passes(
        MakeNameForTempProduct(module->getModuleIdentifier(), "-nvptx.dummy"),
        "", false);
    codegen_passes.add(new llvm::TargetLibraryInfoWrapperPass(
        llvm::Triple(module->getTargetTriple())));

    target_machine->addPassesToEmitFile(codegen_passes, pstream, nullptr,
                                        llvm::TargetMachine::CGFT_AssemblyFile);
    codegen_passes.run(*module);
  }

  return ptx;
}

// LLVM has an extensive flags mechanism of its own, which is only accessible
// through the command line. Internal libraries within LLVM register parsers for
// flags, with no other way to configure them except pass these flags.
// To do this programmatically, we invoke ParseCommandLineOptions manually with
// a "fake argv".
// Note: setting flags with this method is stateful, since flags are just
// static globals within LLVM libraries.
void FeedLLVMWithFlags(const std::vector<std::string>& cl_opts) {
  std::vector<const char*> fake_argv = {""};
  for (const std::string& cl_opt : cl_opts) {
    fake_argv.push_back(cl_opt.c_str());
  }
  llvm::cl::ParseCommandLineOptions(fake_argv.size(), &fake_argv[0]);
}

// Returns whether the module could use any device bitcode library functions.
// This function may have false positives -- the module might not use libdevice
// on NVPTX or ROCm-Device-Libs on AMDGPU even if this function returns true.
bool CouldNeedDeviceBitcode(const llvm::Module& module) {
  for (const llvm::Function& function : module.functions()) {
    // This is a conservative approximation -- not all such functions are in
    // libdevice or ROCm-Device-Libs.
    if (!function.isIntrinsic() && function.isDeclaration()) {
      return true;
    }
  }
  return false;
}

// Links the module with a vector of path to bitcode modules.
// The caller must guarantee that the paths exist.
Status LinkWithBitcodeVector(
    llvm::Module* module, const std::vector<std::string>& bitcode_path_vector,
    const std::string& ir_path, const std::string& linked_ir_path,
    const std::string& optimized_ir_path) {
  std::error_code ec;
  std::string error_message;

  for (auto& bitcode_path : bitcode_path_vector) {
    if (!tensorflow::Env::Default()->FileExists(bitcode_path).ok()) {
      LOG(ERROR) << "bitcode module is required by this HLO module but was "
                    "not found at "
                 << bitcode_path;
      return xla::InternalError("bitcode module not found at %s", bitcode_path);
    }
  }

  // Dump LLVM IR.
  std::string ir_string;
  llvm::raw_string_ostream stream(ir_string);
  // llvm::buffer_ostream pstream(stream);
  module->print(stream, nullptr);
  stream.flush();
  ir_string = absl::StrReplaceAll(ir_string, {{"v1024:1024-v2048:2048-n32:64-S32-A5", "v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"}});

  std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
      new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::F_None));
  // module->print(*ir_fs, nullptr);
  *ir_fs << ir_string;
  ir_fs->flush();

  // Locate llvm-link.
  std::string llvmlink_path = tensorflow::io::JoinPath("/opt/rocm", "llvm/bin");
  auto llvmlink_program =
      llvm::sys::findProgramByName("llvm-link", {llvmlink_path});
  if (!llvmlink_program) {
    return xla::InternalError("unable to find llvm-link in PATH: %s",
                              llvmlink_program.getError().message());
  }
  // Setup llvm-link arguments.
  std::vector<llvm::StringRef> llvmlink_args{
      llvm_ir::AsStringRef("llvm-link"),
      llvm_ir::AsStringRef("-o"),
      llvm_ir::AsStringRef(linked_ir_path),
      llvm_ir::AsStringRef("--suppress-warnings"),
  };

  llvmlink_args.push_back(llvm_ir::AsStringRef(ir_path));
  for (auto& bitcode_path : bitcode_path_vector) {
    llvmlink_args.push_back(llvm_ir::AsStringRef(bitcode_path));
  }

  int llvmlink_result = llvm::sys::ExecuteAndWait(
      *llvmlink_program, llvm_ir::AsArrayRef(llvmlink_args), llvm::None, {}, 0,
      0, &error_message);

  if (llvmlink_result) {
    return xla::InternalError("llvm-link execute fail: %s", error_message);
  }

  // Locate opt.
  std::string opt_path = tensorflow::io::JoinPath("/opt/rocm", "llvm/bin");
  auto opt_program = llvm::sys::findProgramByName("opt", {opt_path});
  if (!opt_program) {
    return xla::InternalError("unable to find opt in PATH: %s",
                              opt_program.getError().message());
  }
  std::vector<llvm::StringRef> opt_args{
      llvm_ir::AsStringRef("opt"),
      llvm_ir::AsStringRef("-O3"),
      llvm_ir::AsStringRef("-o"),
      llvm_ir::AsStringRef(optimized_ir_path),
      llvm_ir::AsStringRef(linked_ir_path),
  };

  int opt_result =
      llvm::sys::ExecuteAndWait(*opt_program, llvm_ir::AsArrayRef(opt_args),
                                llvm::None, {}, 0, 0, &error_message);

  if (opt_result) {
    return xla::InternalError("opt execute fail: %s", error_message);
  }

  return Status::OK();
}

// Links libdevice into the given module if the module needs libdevice.
Status LinkLibdeviceIfNecessary(llvm::Module* module,
                                std::pair<int, int> compute_capability,
                                const std::string& libdevice_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return Status::OK();
  }

  // CUDA 9+ uses a single libdevice file for all devices, and we don't support
  // older CUDAs.
  std::string libdevice_path =
      tensorflow::io::JoinPath(libdevice_dir_path, "libdevice.10.bc");
  if (!tensorflow::Env::Default()->FileExists(libdevice_path).ok()) {
    LOG(WARNING)
        << "libdevice is required by this HLO module but was not found at "
        << libdevice_path;
    return xla::InternalError("libdevice not found at %s", libdevice_path);
  }

  VLOG(1) << "Linking with libdevice from: " << libdevice_path;
  return LinkWithBitcodeVector(module, {libdevice_path}, "", "", "");
}

Status NVPTXTargetModuleLinker(llvm::Module* module, GpuVersion gpu_version,
                               const HloModuleConfig& hlo_module_config,
                               const std::string& device_bitcode_dir_path,
                               const std::string& ir_path,
                               const std::string& linked_ir_path,
                               const std::string& optimized_ir_path) {
  // Link the input module with libdevice, to pull in implementations of some
  // builtins.
  auto compute_capability = absl::get_if<std::pair<int, int>>(&gpu_version);
  if (!compute_capability) {
    return xla::InternalError("Incompatible compute capability was specified.");
  }
  TF_RETURN_IF_ERROR(LinkLibdeviceIfNecessary(module, *compute_capability,
                                              device_bitcode_dir_path));

  // Set the flush-denormals-to-zero flag on the module so the NVVM reflect
  // pass can access it.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        hlo_module_config.debug_options().xla_gpu_ftz());

  // If ftz is enabled, set it as an attribute on every function in the
  // module.
  if (hlo_module_config.debug_options().xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("nvptx-f32ftz", "true");
    }
  }

  return Status::OK();
}

std::unique_ptr<llvm::TargetMachine> NVPTXGetTargetMachine(
    llvm::Triple target_triple, std::pair<int, int> compute_capability,
    const HloModuleConfig& hlo_module_config) {
  // Figure out the exact name of the processor as known to the NVPTX backend
  // from the gpu_architecture flag.
  return GetTargetMachine(target_triple, GetSmName(compute_capability),
                          hlo_module_config, "+ptx60");
}

using TargetModuleLinker = std::function<Status(
    llvm::Module*, GpuVersion, const HloModuleConfig&, const std::string&,
    const std::string&, const std::string&, const std::string&)>;

Status LinkAndOptimizeModule(llvm::Module* module, GpuVersion gpu_version,
                             const HloModuleConfig& hlo_module_config,
                             const std::string& device_bitcode_dir_path,
                             TargetModuleLinker module_linker,
                             llvm::Triple default_target_triple,
                             llvm::TargetMachine* target_machine,
                             int inline_threshold, const std::string& ir_path,
                             const std::string& linked_ir_path,
                             const std::string& optimized_ir_path) {
  TF_RETURN_IF_ERROR(module_linker(module, gpu_version, hlo_module_config,
                                   device_bitcode_dir_path, ir_path,
                                   linked_ir_path, optimized_ir_path));

  return Status::OK();
}

// One-time module initializer.
// Must be called only once -- DO NOT CALL DIRECTLY.
void NVPTXBackendInit(const HloModuleConfig& hlo_module_config) {
  // Feed all customized flags here, so we can override them with llvm_cl_opts
  // without redeploy the compiler for development purpose.

  // This flag tunes a threshold in branch folding. The default threshold, which
  // is one, is not suitable for CUDA programs where branches are more expensive
  // than for CPU programs. Setting the threshold to 2 improves the latency of
  // TwoDPatchDotProductKernel_IND_3_ND_48 by over 5%, and does not affect the
  // latency of other benchmarks so far.
  //
  // I also tried setting this threshold to other values:
  // * 3-6 gives similar results as 2;
  // * >6 start hurting the performance of at least dot product kernels.
  //
  // TODO(jingyue): The current threshold only considers the number of IR
  // instructions which do not accurately reflect the true cost. We need a
  // better cost model.
  FeedLLVMWithFlags({"-bonus-inst-threshold=2"});
  // Increase limit when scanning memory dependencies.  This helps to reduce
  // more redundant load instructions.
  //
  // The specific value is currently large enough for s3d in shoc benchmark,
  // which contains a lot of load instructions and many arithmetic instructions
  // between those loads.
  FeedLLVMWithFlags({"-memdep-block-scan-limit=500"});

  // Use div.full -- it matters for some float-division heavy benchmarks.
  // Using div.approx produces incorrect result for float32(max)/float32(max).
  FeedLLVMWithFlags({"-nvptx-prec-divf32=1"});

  llvm_ir::InitializeLLVMCommandLineOptions(hlo_module_config);

  // Initialize the NVPTX target; it's the only target we link with, so call its
  // specific initialization functions instead of the catch-all InitializeAll*.
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  // Initialize the LLVM optimization passes.
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

}  // namespace

namespace nvptx {

StatusOr<string> CompileToPtx(llvm::Module* module, GpuVersion gpu_version,
                              const HloModuleConfig& hlo_module_config,
                              const std::string& libdevice_dir_path) {
  static std::once_flag backend_init_flag;
  std::call_once(backend_init_flag, NVPTXBackendInit, hlo_module_config);

  std::string ptx;
  std::unique_ptr<llvm::TargetMachine> target_machine;
  {
    tensorflow::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR:", module->getName().str()); },
        tensorflow::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    // If the module has no functions or globals, there's nothing to compile.
    // Just return an empty string.
    if (module->empty() && module->global_empty()) {
      VLOG(2) << "Module '" << module->getName().str()
              << "' is empty. Skipping compilation.";
      return std::string();
    }

    auto compute_capability = absl::get_if<std::pair<int, int>>(&gpu_version);
    if (!compute_capability) {
      return xla::InternalError(
          "Incompatible compute capability was specified.");
    }

    llvm::Triple default_target_triple("nvptx64-unknown-unknown");
    // Construct LLVM TargetMachine for NVPTX.
    std::unique_ptr<llvm::TargetMachine> target_machine = NVPTXGetTargetMachine(
        default_target_triple, *compute_capability, hlo_module_config);

    // Link with libdeivce, and optimize the LLVM module.
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, hlo_module_config, libdevice_dir_path,
        NVPTXTargetModuleLinker, default_target_triple, target_machine.get(),
        kDefaultInlineThreshold, "", "", ""));

    // Lower optimized LLVM module to PTX.
    ptx = EmitModuleToPTX(module, target_machine.get());
  }
  return ptx;
}

}  // namespace nvptx

namespace {
static std::string hsaco_cache_dir_;
static absl::flat_hash_map<std::string, std::vector<uint8>> hsaco_cache_;

static void InitHsacoCacheDir() {
  static absl::once_flag init_once;
  absl::call_once(init_once, [] {
    auto env = tensorflow::Env::Default();
    tensorflow::ReadStringFromEnvVar("TF_XLA_HSACO_CACHE_DIR", "/tmp",
                                     &hsaco_cache_dir_);
    if (hsaco_cache_dir_.empty()) {
      LOG(INFO) << "Will not cache XLA HSACOs. "
                << "This line is logged at most "
                << "once for the lifetime of the process.";
    } else {
      if (!env->IsDirectory(hsaco_cache_dir_).ok()) {
        env->CreateDir(hsaco_cache_dir_);
      }
      LOG(INFO) << "Cache XLA HSACOs in " << hsaco_cache_dir_ << ". "
                << "This line is logged at most "
                << "once for the lifetime of the process.";
    }
  });
}

// Gets the ROCm-Device-Libs filenames for a particular AMDGPU version.
static std::vector<std::string> GetROCDLPaths(
    int amdgpu_version, const std::string& rocdl_dir_path) {
  // AMDGPU version-neutral bitcodes.
  static std::vector<string>* rocdl_filenames = new std::vector<string>(
      {"opencl.bc", "ocml.bc", "ockl.bc", "oclc_finite_only_off.bc",
       "oclc_daz_opt_off.bc", "oclc_correctly_rounded_sqrt_on.bc",
       "oclc_unsafe_math_off.bc", "oclc_wavefrontsize64_on.bc", "hip.bc",
       "oclc_abi_version_500.bc"});

  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  for (auto& filename : *rocdl_filenames) {
    result.push_back(tensorflow::io::JoinPath(rocdl_dir_path, filename));
  }

  // Add AMDGPU version-specific bitcodes.
  result.push_back(tensorflow::io::JoinPath(
      rocdl_dir_path,
      absl::StrCat("oclc_isa_version_", amdgpu_version, ".bc")));
  return result;
}

Status ReadHsaco(std::string hsaco_path, std::vector<uint8>& hsaco) {
  auto it = hsaco_cache_.find(hsaco_path);
  if (it != hsaco_cache_.end()) {
    VLOG(1) << "Hsaco cache hit in memory " << hsaco_path;
    hsaco = it->second;
    return Status::OK();
  }
  if (tensorflow::Env::Default()->FileExists(hsaco_path).ok()) {
    VLOG(1) << "Hsaco cache hit in file " << hsaco_path;
    std::ifstream hsaco_file(hsaco_path, std::ios::binary | std::ios::ate);
    std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();
    hsaco = std::vector<uint8>(hsaco_file_size);
    hsaco_file.seekg(0, std::ios::beg);
    hsaco_file.read(reinterpret_cast<char*>(&hsaco[0]), hsaco_file_size);
    hsaco_cache_.emplace(hsaco_path, hsaco);
    return Status::OK();
  }
  return xla::InternalErrorStrCat("Can't find Hsaco: ", hsaco_path);
}

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the AMDGPU target.
StatusOr<std::vector<uint8>> EmitModuleToHsaco(
    Module* module, llvm::TargetMachine* target_machine,
    const std::string& optimized_ir_path, const std::string& isabin_path,
    const std::string& hsaco_path, int amdgpu_version) {
  std::string error_message;

  // Locate llc.
  std::string llc_path = tensorflow::io::JoinPath("/opt/rocm", "llvm/bin");
  auto llc_program = llvm::sys::findProgramByName("llc", {llc_path});
  if (!llc_program) {
    return xla::InternalError("unable to find llc in PATH: %s",
                              llc_program.getError().message());
  }
  std::vector<llvm::StringRef> llc_args{
      llvm_ir::AsStringRef("llc"),
      llvm_ir::AsStringRef("-march=amdgcn"),
      llvm_ir::AsStringRef(
          absl::StrCat("-mcpu=gfx", std::to_string(amdgpu_version))),
      llvm_ir::AsStringRef("--amdgpu-kernarg-preload-count=16"),
      llvm_ir::AsStringRef("-filetype=obj"),
      llvm_ir::AsStringRef("-o"),
      llvm_ir::AsStringRef(isabin_path),
      llvm_ir::AsStringRef(optimized_ir_path),
  };

  int llc_result =
      llvm::sys::ExecuteAndWait(*llc_program, llvm_ir::AsArrayRef(llc_args),
                                llvm::None, {}, 0, 0, &error_message);

  if (llc_result) {
    return xla::InternalError("llc execute fail: %s", error_message);
  }

  // Locate lld.
  // TODO(whchung@gmail.com): change to tensorflow::ROCmRoot() after
  // ROCm-Device-Libs PR.
  std::string lld_path_1 = tensorflow::io::JoinPath("/opt/rocm", "hcc/bin");
  std::string lld_path_2 = tensorflow::io::JoinPath("/opt/rocm", "llvm/bin");
  auto lld_program =
      llvm::sys::findProgramByName("ld.lld", {lld_path_1, lld_path_2});
  if (!lld_program) {
    return xla::InternalError("unable to find ld.lld in PATH: %s",
                              lld_program.getError().message());
  }
  std::vector<llvm::StringRef> lld_args{
      llvm_ir::AsStringRef("ld.lld"),    llvm_ir::AsStringRef("-flavor"),
      llvm_ir::AsStringRef("gnu"),       llvm_ir::AsStringRef("-shared"),
      llvm_ir::AsStringRef(isabin_path), llvm_ir::AsStringRef("-o"),
      llvm_ir::AsStringRef(hsaco_path),
  };

  int lld_result =
      llvm::sys::ExecuteAndWait(*lld_program, llvm_ir::AsArrayRef(lld_args),
                                llvm::None, {}, 0, 0, &error_message);

  if (lld_result) {
    return xla::InternalError("ld.lld execute fail: %s", error_message);
  }

  // Read HSACO.
  std::vector<uint8> hsaco;
  ReadHsaco(hsaco_path, hsaco);
  return hsaco;
}

// Links ROCm-Device-Libs into the given module.
Status LinkROCDLIfNecessary(llvm::Module* module, int amdgpu_version,
                            const std::string& rocdl_dir_path,
                            const std::string& ir_path,
                            const std::string& linked_ir_path,
                            const std::string& optimized_ir_path) {
  return LinkWithBitcodeVector(module,
                               GetROCDLPaths(amdgpu_version, rocdl_dir_path),
                               ir_path, linked_ir_path, optimized_ir_path);
}

Status AMDGPUTargetModuleLinker(llvm::Module* module, GpuVersion gpu_version,
                                const HloModuleConfig& hlo_module_config,
                                const std::string& device_bitcode_dir_path,
                                const std::string& ir_path,
                                const std::string& linked_ir_path,
                                const std::string& optimized_ir_path) {
  // Link the input module with ROCDL.
  auto amdgpu_version = absl::get_if<int>(&gpu_version);
  if (!amdgpu_version) {
    return xla::InternalError(
        "Incompatible AMD GCN ISA version was specified.");
  }
  TF_RETURN_IF_ERROR(LinkROCDLIfNecessary(module, *amdgpu_version,
                                          device_bitcode_dir_path, ir_path,
                                          linked_ir_path, optimized_ir_path));

  return Status::OK();
}

std::unique_ptr<llvm::TargetMachine> AMDGPUGetTargetMachine(
    llvm::Triple target_triple, int amdgpu_version,
    const HloModuleConfig& hlo_module_config) {
  return {};
  // GetTargetMachine(target_triple, absl::StrCat("gfx", amdgpu_version),
  //                       hlo_module_config, "+code-object-v3");
}

void AMDGPUBackendInit(const HloModuleConfig& hlo_module_config) {
  llvm_ir::InitializeLLVMCommandLineOptions(hlo_module_config);

  // Initialize the AMDGPU target; it's the only target we link with, so call
  // its specific initialization functions instead of the catch-all
  // InitializeAll*.
#if TENSORFLOW_USE_ROCM
  InitHsacoCacheDir();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
#endif

  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

}  // namespace

namespace amdgpu {

StatusOr<std::vector<uint8>> CompileToHsaco(
    llvm::Module* module, GpuVersion gpu_version,
    const HloModuleConfig& hlo_module_config,
    const std::string& rocdl_dir_path) {
  static std::once_flag backend_init_flag;
  std::call_once(backend_init_flag, AMDGPUBackendInit, hlo_module_config);

  std::vector<uint8> hsaco;
  std::unique_ptr<llvm::TargetMachine> target_machine;

  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << *module;

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

  {
    tensorflow::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR", module->getName().str()); },
        tensorflow::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    auto amdgpu_version = absl::get_if<int>(&gpu_version);
    if (!amdgpu_version) {
      return xla::InternalError(
          "Incompatible AMD GCN ISA version was specified.");
    }

    std::string hsaco_filename =
        absl::StrCat(std::hash<std::string>{}(str), ".gfx",
                     std::to_string(*amdgpu_version), ".hsaco");
    std::string hsaco_path =
        tensorflow::io::JoinPath(hsaco_cache_dir_, hsaco_filename);

    if (ReadHsaco(hsaco_path, hsaco).ok()) {
      VLOG(1) << "HSACO cache hit";
      return hsaco;
    }
    VLOG(1) << "HSACO cache miss";

    llvm::Triple default_target_triple("amdgcn--amdhsa-amdgiz");
    // Construct LLVM TargetMachine for AMDGPU.
    std::unique_ptr<llvm::TargetMachine> target_machine =
        AMDGPUGetTargetMachine(default_target_triple, *amdgpu_version,
                               hlo_module_config);

    auto* env = tensorflow::Env::Default();
    // Prepare filenames for all stages of compilation:
    // IR, binary ISA, and HSACO.
    std::string module_path;
    if (!env->LocalTempFilename(&module_path)) {
      return xla::InternalError(
          "Could not get temporary filenames for modules.");
    }
    std::string ir_path = absl::StrCat(module_path, ".ll");

    std::string linked_ir_path = absl::StrCat(module_path, "-linked.ll");

    std::string optimized_ir_path = absl::StrCat(module_path, "-opt.ll");

    std::string isabin_path = absl::StrCat(module_path, ".o");

    // Link with ROCm-Device-Libs, and optimize the LLVM module.
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, hlo_module_config, rocdl_dir_path,
        AMDGPUTargetModuleLinker, default_target_triple, target_machine.get(),
        kAMDGPUInlineThreshold, ir_path, linked_ir_path, optimized_ir_path));

    // Lower optimized LLVM module to HSA code object.
    TF_ASSIGN_OR_RETURN(hsaco, EmitModuleToHsaco(module, target_machine.get(),
                                                 optimized_ir_path, isabin_path,
                                                 hsaco_path, *amdgpu_version));

    std::async(
        std::launch::async,
        [](std::vector<std::string> files) {
          for (auto& file : files) {
            tensorflow::Env::Default()->DeleteFile(file);
          }
        },
        std::vector<std::string>{ir_path, linked_ir_path, optimized_ir_path,
                                 isabin_path});
  }
  return hsaco;
}

}  // namespace amdgpu

}  // namespace gpu
}  // namespace xla