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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/utils.h"
#include "tensorflow/compiler/xla/service/gpu/metrics.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_command_line_options.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/cuda_libdevice_path.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/random.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"
#include "tensorflow/tsl/util/env_var.h"


#if !defined(PLATFORM_GOOGLE) && TENSORFLOW_USE_ROCM
#include "tensorflow/tsl/platform/rocm_rocdl_path.h"
#include "rocm/rocm_config.h"
#endif

namespace xla {
namespace gpu {
namespace {

static llvm::codegen::RegisterCodeGenFlags CGF;

// Inline threshold value to use in LLVM AMDGPU backend.
const int kAMDGPUInlineThreshold = 0x100000;

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

// Gets the GPU name as it's known to LLVM for a given compute
// capability.  If we see an unrecognized compute capability, we
// return the highest one that is known and below the selected device.
static std::string GetSmName(se::CudaComputeCapability compute_capability) {
  int compute_capability_version =
      compute_capability.major * 10 + compute_capability.minor;
  int sm_version = 30;
  // If the current compute capability isn't known, fallback to the
  // most recent version before it.
  int supported_versions[] = {90, 89, 87, 86, 80, 75, 72, 70, 62,
                              61, 60, 53, 52, 50, 37, 35, 32, 30};
  for (int v : supported_versions) {
    if (v <= compute_capability_version) {
      sm_version = v;
      break;
    }
  }

  // If the current CC isn't supported by LLVM and it is newer then
  // the max supported LLVM version, do not warn about it. The end
  // user can't do anything about this. E.g., PTX compiled for SM75 will
  // run on SM80 too.
  if (sm_version != compute_capability_version &&
      compute_capability_version < supported_versions[0]) {
    LOG(WARNING) << "Unknown compute capability "
                 << compute_capability.ToString()
                 << ". Defaulting to telling LLVM that we're compiling for sm_"
                 << sm_version;
  }
  return absl::StrCat("sm_", sm_version);
}

// Convenience function for producing a name of a temporary compilation product
// from the input filename.
std::string MakeNameForTempProduct(absl::string_view input_filename,
                                   absl::string_view extension) {
  return ReplaceFilenameExtension(tsl::io::Basename(input_filename), extension);
}

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(llvm::PassRegistry* pass_registry) {
  llvm::initializeCore(*pass_registry);
  llvm::initializeCodeGen(*pass_registry);
  llvm::initializeScalarOpts(*pass_registry);
  llvm::initializeVectorization(*pass_registry);
  llvm::initializeIPO(*pass_registry);
  llvm::initializeAnalysis(*pass_registry);
  llvm::initializeTransformUtils(*pass_registry);
  llvm::initializeInstCombine(*pass_registry);
  llvm::initializeTarget(*pass_registry);
  llvm::initializeCodeGenPreparePass(*pass_registry);
}

// Returns the TargetMachine, given a triple.
std::unique_ptr<llvm::TargetMachine> GetTargetMachine(
    llvm::Triple triple, absl::string_view cpu_name,
    const HloModuleConfig& hlo_module_config, absl::string_view feature_str) {
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("", triple, error);
  if (target == nullptr) {
    LOG(FATAL) << "Unable to find Target for triple '" << triple.str() << "'"
               << " -- " << error;
    return nullptr;
  }

  llvm::TargetOptions target_options =
      llvm::codegen::InitTargetOptionsFromCodeGenFlags(llvm::Triple());

  // Set the verbose assembly options.
  target_options.MCOptions.AsmVerbose = false;

  // The selection of codegen optimization level is copied from function
  // GetCodeGenOptLevel in //third_party/llvm/llvm/tools/opt/opt.cpp.
  llvm::CodeGenOpt::Level codegen_opt_level;
  switch (hlo_module_config.debug_options().xla_backend_optimization_level()) {
    case 1:
      codegen_opt_level = llvm::CodeGenOpt::Less;
      break;
    case 2:
      codegen_opt_level = llvm::CodeGenOpt::Default;
      break;
    case 3:
      codegen_opt_level = llvm::CodeGenOpt::Aggressive;
      break;
    default:
      codegen_opt_level = llvm::CodeGenOpt::None;
  }
  return absl::WrapUnique(target->createTargetMachine(
      triple.str(), llvm_ir::AsStringRef(cpu_name),
      llvm_ir::AsStringRef(feature_str), target_options,
      llvm::codegen::getExplicitRelocModel(),
      llvm::codegen::getExplicitCodeModel(), codegen_opt_level));
}

// Emits the given module to PTX. target_machine is an initialized TargetMachine
// for the NVPTX target.
std::string EmitModuleToPTX(llvm::Module* module,
                            llvm::TargetMachine* target_machine) {
  std::string ptx;
  llvm::raw_string_ostream stream(ptx);
  llvm::buffer_ostream pstream(stream);
  llvm::legacy::PassManager pm;
  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  target_machine->addPassesToEmitFile(pm, pstream, nullptr,
                                      llvm::CGFT_AssemblyFile);
  pm.run(*module);
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
bool CouldNeedDeviceBitcode(const llvm::Module& module) {
  for (const llvm::Function& function : module.functions()) {
    // The list of prefixes should be in sync with library functions used in
    // target_util.cc.
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().startswith("__nv_") ||
         function.getName().startswith("__ocml_") ||
         function.getName().startswith("__ockl_"))) {
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
    if (!tsl::Env::Default()->FileExists(bitcode_path).ok()) {
      LOG(ERROR) << "bitcode module is required by this HLO module but was "
                    "not found at "
                 << bitcode_path;
      return xla::InternalError("bitcode module not found at %s", bitcode_path);
    }
  }

  // Dump LLVM IR.
  std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
      new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::OF_None));
  module->print(*ir_fs, nullptr);
  ir_fs->flush();

  // Locate llvm-link.
  std::string llvmlink_path = tsl::io::JoinPath(tsl::RocmRoot(), "llvm/bin");
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
  };

  llvmlink_args.push_back(llvm_ir::AsStringRef(ir_path));
  for (auto& bitcode_path : bitcode_path_vector) {
    llvmlink_args.push_back(llvm_ir::AsStringRef(bitcode_path));
  }

  int llvmlink_result = llvm::sys::ExecuteAndWait(
      *llvmlink_program, llvm_ir::AsArrayRef(llvmlink_args), std::nullopt, {},
      0, 0, &error_message);

  if (llvmlink_result) {
    return xla::InternalError("llvm-link execute fail: %s", error_message);
  }

  // Locate opt.
  std::string opt_path = tsl::io::JoinPath(tsl::RocmRoot(), "llvm/bin");
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
                                std::nullopt, {}, 0, 0, &error_message);

  if (opt_result) {
    return xla::InternalError("opt execute fail: %s", error_message);
  }
  return OkStatus();
}

// Links libdevice into the given module if the module needs libdevice.
Status LinkLibdeviceIfNecessary(llvm::Module* module,
                                const std::string& libdevice_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return OkStatus();
  }

  // CUDA 9+ uses a single libdevice file for all devices, and we don't support
  // older CUDAs.
  std::string libdevice_path =
      tsl::io::JoinPath(libdevice_dir_path, "libdevice.10.bc");
  if (!tsl::Env::Default()->FileExists(libdevice_path).ok()) {
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
  TF_RETURN_IF_ERROR(LinkLibdeviceIfNecessary(module, device_bitcode_dir_path));

  // Set the flush-denormals-to-zero flag on the module so the NVVM reflect pass
  // can access it.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        hlo_module_config.debug_options().xla_gpu_ftz());

  // If ftz is enabled, set it as an attribute on every function in the module.
  if (hlo_module_config.debug_options().xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }

  return OkStatus();
}

std::unique_ptr<llvm::TargetMachine> NVPTXGetTargetMachine(
    llvm::Triple target_triple, se::CudaComputeCapability compute_capability,
    const HloModuleConfig& hlo_module_config) {
  // TODO(b/266678775): Make it always PTX 7.1 as soon as TF driver requirements
  // are updated.
  const std::string ptx_ver =
      hlo_module_config.debug_options().xla_gpu_enable_triton_gemm() ? "+ptx71"
                                                                     : "+ptx60";
  // Figure out the exact name of the processor as known to the NVPTX backend
  // from the gpu_architecture flag.
  return GetTargetMachine(target_triple, GetSmName(compute_capability),
                          hlo_module_config, ptx_ver);
}

using TargetModuleLinker = std::function<Status(
    llvm::Module*, GpuVersion, const HloModuleConfig&, const std::string&,
    const std::string&, const std::string&, const std::string&)>;

void DumpModule(const std::string output_filename, const llvm::Module* module) {
  std::error_code ec;
  auto out = std::make_unique<llvm::raw_fd_ostream>(
      llvm::StringRef(output_filename), ec, llvm::sys::fs::OF_None);
  if (ec) {
    LOG(FATAL) << "Unable to open " << output_filename
               << " to dump LLVM IR: " << ec.message();
    return;
  }
  module->print(*out, /*AAW=*/nullptr);
  out->close();
}

const llvm::Module* GetModule(llvm::Any IR) {
  if (llvm::any_isa<const llvm::Module*>(IR))
    return llvm::any_cast<const llvm::Module*>(IR);

  if (llvm::any_isa<const llvm::Function*>(IR)) {
    const llvm::Function* F = llvm::any_cast<const llvm::Function*>(IR);
    return F->getParent();
  }

  if (llvm::any_isa<const llvm::LazyCallGraph::SCC*>(IR)) {
    const llvm::LazyCallGraph::SCC* C =
        llvm::any_cast<const llvm::LazyCallGraph::SCC*>(IR);
    return C->begin()->getFunction().getParent();
  }

  if (llvm::any_isa<const llvm::Loop*>(IR)) {
    const llvm::Loop* L = llvm::any_cast<const llvm::Loop*>(IR);
    const llvm::Function* F = L->getHeader()->getParent();
    return F->getParent();
  }

  return nullptr;
}

auto DumpCallbackForModule(std::string module_identifier,
                           std::string outputs_dir) {
  int i = 0;
  return [=](llvm::StringRef pass, llvm::Any ir) mutable {
    const llvm::Module* module = GetModule(ir);
    if (!module) {
      return;
    }

    const std::string basename = ReplaceFilenameExtension(
        absl::string_view(tsl::io::Basename(module_identifier)),
        absl::StrFormat("pass-%02d.before.%s.ll", i++,
                        absl::string_view(pass.str())));
    DumpModule(tsl::io::JoinPath(outputs_dir, basename), module);
  };
}

Status LinkAndOptimizeModule(llvm::Module* module, GpuVersion gpu_version,
                             const HloModuleConfig& hlo_module_config,
                             const std::string& device_bitcode_dir_path,
                             TargetModuleLinker module_linker,
                             llvm::Triple default_target_triple,
                             llvm::TargetMachine* target_machine,
                             int inline_threshold, const std::string& ir_path,
                             const std::string& linked_ir_path,
                             const std::string& optimized_ir_path) {
  return module_linker(module, gpu_version, hlo_module_config,
                       device_bitcode_dir_path, ir_path, linked_ir_path,
                       optimized_ir_path);
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

  // Use div.full -- it matters for some float-division heavy benchmarks.
  // Using div.approx produces incorrect result for float32(max)/float32(max).
  FeedLLVMWithFlags({"-nvptx-prec-divf32=1"});

  // SLPVectorizer is useful (vectorizes f16x2 ops) but slow.  Most of the
  // slowness appears to be in trying to form horizontal reductions, which don't
  // exist in PTX *anyway*.  Disable these.  While we're here, tweak
  // SLPVectorizer so it doesn't try to create large vectors -- f16x2 are the
  // only vectors supported in PTX.
  FeedLLVMWithFlags({
      "-slp-vectorize-hor=false",
      "-slp-max-reg-size=32",
  });

  llvm_ir::InitializeLLVMCommandLineOptions(
      hlo_module_config.debug_options().xla_backend_extra_options());

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

std::string CantFindCudaMessage(absl::string_view msg,
                                absl::string_view xla_gpu_cuda_data_dir) {
  return absl::StrCat(
      msg, "\nSearched for CUDA in the following directories:\n  ",
      absl::StrJoin(tsl::CandidateCudaRoots(std::string{xla_gpu_cuda_data_dir}),
                    "\n  "),
      "\nYou can choose the search directory by setting xla_gpu_cuda_data_dir "
      "in HloModule's DebugOptions.  For most apps, setting the environment "
      "variable XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda will work.");
}

static std::string GetLibdeviceDir(absl::string_view xla_gpu_cuda_data_dir) {
  for (const std::string& cuda_root :
       tsl::CandidateCudaRoots(std::string{xla_gpu_cuda_data_dir})) {
    std::string libdevice_dir =
        tsl::io::JoinPath(cuda_root, "nvvm", "libdevice");
    VLOG(2) << "Looking for libdevice at " << libdevice_dir;
    if (tsl::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << libdevice_dir;
      return libdevice_dir;
    }
  }
  LOG(WARNING) << CantFindCudaMessage(
      "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may "
      "result in compilation or runtime failures, if the program we try to run "
      "uses routines from libdevice.",
      xla_gpu_cuda_data_dir);

  // GetCudaRootCandidates always includes ".", but if everything fails, we
  // return it anyway.  Better than returning the empty string.
  return ".";
}

StatusOr<std::string> CompileToPtx(
    llvm::Module* module, GpuVersion gpu_version,
    const HloModuleConfig& hlo_module_config,
    std::function<void(llvm::TargetMachine*)> configure_target) {
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, NVPTXBackendInit, hlo_module_config);

  absl::string_view xla_gpu_cuda_data_dir =
      hlo_module_config.debug_options().xla_gpu_cuda_data_dir();

  static absl::Mutex libdevice_cache_mu(absl::kConstInit);
  static auto& libdevice_dir_path_cache ABSL_GUARDED_BY(libdevice_cache_mu) =
      *new absl::flat_hash_map<std::string, std::string>();
  std::string libdevice_dir_path = [&] {
    absl::MutexLock l(&libdevice_cache_mu);
    auto it = libdevice_dir_path_cache.find(xla_gpu_cuda_data_dir);
    if (it != libdevice_dir_path_cache.end()) {
      return it->second;
    }
    auto [it2, inserted] = libdevice_dir_path_cache.emplace(
        xla_gpu_cuda_data_dir, GetLibdeviceDir(xla_gpu_cuda_data_dir));
    return it2->second;
  }();

  std::string ptx;
  std::unique_ptr<llvm::TargetMachine> target_machine;
  {
    tsl::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR:", module->getName().str()); },
        tsl::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    // If the module has no functions or globals, there's nothing to compile.
    // Just return an empty string.
    if (module->empty() && module->global_empty()) {
      VLOG(2) << "Module '" << module->getName().str()
              << "' is empty. Skipping compilation.";
      return std::string();
    }

    auto compute_capability =
        std::get_if<se::CudaComputeCapability>(&gpu_version);
    if (!compute_capability) {
      return xla::InternalError(
          "Incompatible compute capability was specified.");
    }

    llvm::Triple default_target_triple("nvptx64-unknown-unknown");
    // Construct LLVM TargetMachine for NVPTX.
    std::unique_ptr<llvm::TargetMachine> target_machine = NVPTXGetTargetMachine(
        default_target_triple, *compute_capability, hlo_module_config);

    // Apply target machine configuration from call-back if available.
    if (configure_target) {
      configure_target(target_machine.get());
    }

    uint64_t start_usecs = tsl::Env::Default()->NowMicros();

    // Link with libdevice, and optimize the LLVM module.
    TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
        module, gpu_version, hlo_module_config, libdevice_dir_path,
        NVPTXTargetModuleLinker, default_target_triple, target_machine.get(),
        kDefaultInlineThreshold, "", "", ""));

    uint64_t end_usecs = tsl::Env::Default()->NowMicros();
    RecordLlvmPassesDuration(end_usecs - start_usecs);

    start_usecs = tsl::Env::Default()->NowMicros();

    // Lower optimized LLVM module to PTX.
    ptx = EmitModuleToPTX(module, target_machine.get());

    end_usecs = tsl::Env::Default()->NowMicros();
    RecordLlvmToPtxDuration(end_usecs - start_usecs);
  }
  return ptx;
}

}  // namespace nvptx

namespace {
static std::string hsaco_cache_dir_;
static std::mutex hsaco_cache_mutex_;
static absl::flat_hash_map<std::string, std::vector<uint8_t>> hsaco_cache_;

static void InitHsacoCacheDir() {
  static absl::once_flag init_once;
  absl::call_once(init_once, [] {
    auto env = tsl::Env::Default();
    tsl::ReadStringFromEnvVar("TF_XLA_HSACO_CACHE_DIR", "/tmp",
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
std::vector<std::string> GetROCDLPaths(std::string gcn_arch_name,
                                       const std::string& rocdl_dir_path) {
  // AMDGPU version-neutral bitcodes.
  static std::vector<std::string>* rocdl_filenames =
      new std::vector<std::string>(
          {"opencl.bc", "ocml.bc", "ockl.bc", "oclc_finite_only_off.bc",
           "oclc_daz_opt_off.bc", "oclc_correctly_rounded_sqrt_on.bc",
           "oclc_unsafe_math_off.bc", "oclc_wavefrontsize64_on.bc", "hip.bc",
           "oclc_abi_version_500.bc"});

  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  result.reserve(rocdl_filenames->size() + 1);
  for (auto& filename : *rocdl_filenames) {
    result.push_back(tsl::io::JoinPath(rocdl_dir_path, filename));
  }

  // Add AMDGPU version-specific bitcodes.
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::string amdgpu_version = gcn_arch_name;
  if (!tokens.empty() && tokens[0].size() >= 3) {
    amdgpu_version = tokens[0].substr(3);
  }
  result.push_back(tsl::io::JoinPath(
      rocdl_dir_path,
      absl::StrCat("oclc_isa_version_", amdgpu_version, ".bc")));
  return result;
}

Status ReadHsaco(std::string hsaco_path, std::vector<uint8_t>& hsaco) {
  std::lock_guard<std::mutex> lg(hsaco_cache_mutex_);
  auto it = hsaco_cache_.find(hsaco_path);
  if (it != hsaco_cache_.end()) {
    VLOG(1) << "Hsaco cache hit in memory " << hsaco_path;
    hsaco = it->second;
    return OkStatus();
  }
  if (tsl::Env::Default()->FileExists(hsaco_path).ok()) {
    VLOG(1) << "Hsaco cache hit in file " << hsaco_path;
    std::ifstream hsaco_file(hsaco_path, std::ios::binary | std::ios::ate);
    std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();
    hsaco = std::vector<uint8_t>(hsaco_file_size);
    hsaco_file.seekg(0, std::ios::beg);
    hsaco_file.read(reinterpret_cast<char*>(&hsaco[0]), hsaco_file_size);
    hsaco_cache_.emplace(hsaco_path, hsaco);
    return OkStatus();
  }
  return xla::InternalErrorStrCat("Can't find Hsaco: ", hsaco_path);
}

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the AMDGPU target.
StatusOr<std::vector<uint8_t>> EmitModuleToHsaco(
    llvm::Module* module, llvm::TargetMachine* target_machine,
    const std::string& optimized_ir_path, const std::string& isabin_path,
    const std::string& hsaco_path, std::string& gcn_arch_name) {
  std::string error_message;
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::string gfx = tokens[0];
  // Locate llc.
  std::string llc_path = tsl::io::JoinPath("/opt/rocm", "llvm/bin");
  auto llc_program = llvm::sys::findProgramByName("llc", {llc_path});
  if (!llc_program) {
    return xla::InternalError("unable to find llc in PATH: %s",
                              llc_program.getError().message());
  }
  std::vector<llvm::StringRef> llc_args{
      llvm_ir::AsStringRef("llc"),
      llvm_ir::AsStringRef("-march=amdgcn"),
      llvm_ir::AsStringRef(
          absl::StrCat("-mcpu=", gfx)),
      llvm_ir::AsStringRef("--amdgpu-kernarg-preload-count=16"),
      llvm_ir::AsStringRef("-filetype=obj"),
      llvm_ir::AsStringRef("-o"),
      llvm_ir::AsStringRef(isabin_path),
      llvm_ir::AsStringRef(optimized_ir_path),
  };

  int llc_result =
      llvm::sys::ExecuteAndWait(*llc_program, llvm_ir::AsArrayRef(llc_args),
                                std::nullopt, {}, 0, 0, &error_message);

  if (llc_result) {
    return xla::InternalError("llc execute fail: %s", error_message);
  }

  // Locate lld.
  // TODO(whchung@gmail.com): change to tensorflow::ROCmRoot() after
  // ROCm-Device-Libs PR.
  std::string lld_path_1 = tsl::io::JoinPath(tsl::RocmRoot(), "hcc/bin");
  std::string lld_path_2 = tsl::io::JoinPath(tsl::RocmRoot(), "llvm/bin");
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
                                std::nullopt, {}, 0, 0, &error_message);

  if (lld_result) {
    return xla::InternalError("ld.lld execute fail: %s", error_message);
  }

  // Read HSACO.
  std::vector<uint8_t> hsaco;
  ReadHsaco(hsaco_path, hsaco);
  return hsaco;
}

// Links ROCm-Device-Libs into the given module if the module needs it.
Status LinkROCDLIfNecessary(llvm::Module* module, std::string gcn_arch_name,
                            const std::string& rocdl_dir_path,
                            const std::string& ir_path,
                            const std::string& linked_ir_path,
                            const std::string& optimized_ir_path) {
  return LinkWithBitcodeVector(module,
                               GetROCDLPaths(gcn_arch_name, rocdl_dir_path),
                               ir_path, linked_ir_path, optimized_ir_path);
}

Status AMDGPUTargetModuleLinker(llvm::Module* module, GpuVersion gpu_version,
                                const HloModuleConfig& hlo_module_config,
                                const std::string& device_bitcode_dir_path,
                                const std::string& ir_path,
                                const std::string& linked_ir_path,
                                const std::string& optimized_ir_path) {
  // Link the input module with ROCDL.

  auto compute_capability =
      std::get_if<se::RocmComputeCapability>(&gpu_version);
  if (!compute_capability) {
    return xla::InternalError("Incompatible compute capability was specified.");
  }

  std::string gcn_arch_name = compute_capability->gcn_arch_name();
  TF_RETURN_IF_ERROR(LinkROCDLIfNecessary(module, gcn_arch_name,
                                          device_bitcode_dir_path, ir_path,
                                          linked_ir_path, optimized_ir_path));

  // For rocm, we always enable flush to zero. (for cuda, this is determined
  // via environemnt variables). This deceision was based on the observation
  // Eugene had that the AMD GPU llvm backend has not picked up the atomic add
  // instructions correctly without ftz enabled. We concluded that this should
  // not has major impact as the hipcc path by default enables flush to zero for
  // compilation.
  for (llvm::Function& fn : *module) {
    // may be necessary for the compiler to generate atomics (confirm!)
    fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    fn.addFnAttr("amdgpu-unsafe-fp-atomics", "true");
  }

  return OkStatus();
}

// The following routine maps a feature token extracted from the
// hipDeviceProp_t::gcnArchName string, and maps it to a valid feature_str
// to be used for creating the AMDGPUTarget.
// This mapping is currently in a state of flux because TF XLA uses its
// own copy of LLVM, which is different from the LLVM version used by
// hipcc/runtime in the ROCm install. Ordinarily this is not a problem,
// but right now, the LLVM version used by hipcc/runtime has "targetID"
// related changes which have not yet been upstreamed (to the LLVM repo)
// When that upstreaming happens (and TF LLVM pointer moves past the
// upstream commit), the following mapping will need to change
std::string MapGCNArchNameTokenToFeatureStr(const std::string& token,
                                            const std::string& gfx) {
  if (token == "sramecc+") {
    return "+sramecc";
  } else if (token == "sramecc-") {
    if (gfx == "gfx90a" || gfx == "gfx940" || gfx == "gfx941" ||
        gfx == "gfx942")
      return "";
    return "-sramecc";
  } else if (token == "xnack+") {
    return "+xnack";
  } else if (token == "xnack-") {
    return "-xnack";
  }
  return "";
}

std::pair<std::string, std::string> GetFeatureStrFromGCNArchName(
    const std::string& gcn_arch_name) {
  std::string feature_str;

  std::string gfx = gcn_arch_name;
  // For ROCm versions 4.0 and greater, we need to specify the correct
  // feature str, based on the underlying GPU HW to get max performance.
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::vector<std::string> mapped_tokens;
  if (tokens.size() > 0) gfx = tokens[0];
  for (auto it = tokens.begin(); it != tokens.end(); it++) {
    // Skip the first token, that is the gfxNNN str
    // The rest of the tokens are the feature/targetid strings
    if (it != tokens.begin()) {
      std::string token(*it);
      std::string mapped_token = MapGCNArchNameTokenToFeatureStr(token, gfx);
      mapped_tokens.push_back(mapped_token);
    }
  }
  feature_str = absl::StrJoin(mapped_tokens, ",");

  return std::make_pair(gfx, feature_str);
}

std::unique_ptr<llvm::TargetMachine> AMDGPUGetTargetMachine(
    llvm::Triple target_triple, GpuVersion gpu_version,
    const HloModuleConfig& hlo_module_config) {
  return {};
}

void AMDGPUBackendInit(const HloModuleConfig& hlo_module_config) {
  llvm_ir::InitializeLLVMCommandLineOptions(
      hlo_module_config.debug_options().xla_backend_extra_options());

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
StatusOr<std::vector<uint8_t>> CompileToHsaco(
    llvm::Module* module, GpuVersion gpu_version,
    const HloModuleConfig& hlo_module_config,
    const std::string& rocdl_dir_path) {
  static std::once_flag backend_init_flag;
  std::call_once(backend_init_flag, AMDGPUBackendInit, hlo_module_config);

  std::vector<uint8_t> hsaco;
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
    tsl::profiler::TraceMe activity(
        [&] { return absl::StrCat("Compiling IR", module->getName().str()); },
        tsl::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    auto compute_capability =
        std::get_if<se::RocmComputeCapability>(&gpu_version);
    if (!compute_capability) {
      return xla::InternalError(
          "Incompatible compute capability was specified.");
    }

    std::string gcn_arch_name = compute_capability->gcn_arch_name();

    std::string hsaco_filename =
        absl::StrCat(std::hash<std::string>{}(str),
                     gcn_arch_name, ".hsaco");
    std::string hsaco_path =
        tsl::io::JoinPath(hsaco_cache_dir_, hsaco_filename);

    if (ReadHsaco(hsaco_path, hsaco).ok()) {
      VLOG(1) << "HSACO cache hit";
      return hsaco;
    }
    VLOG(1) << "HSACO cache miss";

    llvm::Triple default_target_triple("amdgcn--amdhsa-amdgiz");
    // Construct LLVM TargetMachine for AMDGPU.
    std::unique_ptr<llvm::TargetMachine> target_machine =
        AMDGPUGetTargetMachine(default_target_triple, gpu_version,
                               hlo_module_config);

    auto* env = tsl::Env::Default();
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
                                                 hsaco_path, gcn_arch_name));

    std::async(
        std::launch::async,
        [](std::vector<std::string> files) {
          for (auto& file : files) {
            tsl::Env::Default()->DeleteFile(file);
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
