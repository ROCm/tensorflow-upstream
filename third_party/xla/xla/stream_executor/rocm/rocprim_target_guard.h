/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCPRIM_TARGET_GUARD_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCPRIM_TARGET_GUARD_H_

// NOTE: this header pulls in rocPRIM device code, so it may only be included
// from translation units compiled by hipcc (i.e. `rocm_library` targets).

#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rocprim/device/config_types.hpp"

namespace stream_executor::rocm {

// Guards against rocPRIM dispatching a kernel whose code object entry does not
// belong to it.
//
// rocPRIM dispatches its device algorithms through a per-architecture
// `trampoline_kernel`, emitting one instantiation per entry of the algorithm's
// tuned-target list. Only the instantiation matching the code object's own
// compile-time architecture gets a body; every other one compiles to
// `__builtin_unreachable()`, which emits no code at all. Those zero-length
// symbols therefore collapse onto the entry point of the *real* kernel that
// follows them in `.text` -- in a gfx90a build, 688 kernel descriptors share
// just 122 distinct code addresses.
//
// Each trampoline still carries its own 64-byte kernel descriptor, and the
// collapsed ones declare 0 VGPRs and 0 bytes of LDS. So dispatching one runs
// the real kernel's code under a descriptor that grants it no registers and no
// shared memory. It immediately accesses LDS outside its (zero-sized)
// allocation, which raises a memory violation carrying no virtual address:
//
//     Memory access fault by GPU node-N ... on address (nil). Reason: Unknown.
//
// Which trampoline gets launched is decided on the host from an architecture
// read back at runtime, and `rocprim::detail::target(hipStream_t)` silently
// degrades to `target_arch::unknown` when that query fails:
//
//     if (host_target_arch(stream, i) != hipSuccess) { i = target_arch::unknown; }
//
// `unknown` resolves to the generic entry of the tuned-target list, which is a
// collapsed symbol on every architecture that *is* in that list (gfx90a,
// gfx942, ...). On gfx950, which is absent from the list, the generic entry is
// the real body, so the same fallback is harmless -- which is why this only
// bites on MI200/MI300.
//
// Resolving the architecture here first makes that degradation unreachable in
// practice: `rocprim::detail::get_device_arch` caches successful lookups in a
// process-wide, per-device table, so after one success rocPRIM's own lookup is
// a plain atomic load that cannot fail. If it cannot be resolved at all we
// return an error rather than dispatching under a bogus descriptor.
//
// TODO(rocm): drop this once the ROCm distro carries rocPRIM with
// ROCm/rocm-libraries#13311, which stops emitting the collapsed kernels.
inline absl::Status EnsureRocprimTargetResolved(hipStream_t stream) {
  namespace rpd = ::rocprim::detail;
  for (int attempt = 0; attempt < 3; ++attempt) {
    rpd::target_arch arch = rpd::target_arch::unknown;
    if (rpd::host_target_arch(stream, arch) == hipSuccess &&
        arch != rpd::target_arch::unknown) {
      return absl::OkStatus();
    }
  }
  return absl::InternalError(
      "rocPRIM could not determine the GPU architecture for this stream. "
      "Refusing to launch: rocPRIM would dispatch its generic kernel, whose "
      "descriptor grants no registers or shared memory, and the launch would "
      "abort the HSA queue with a memory fault.");
}

}  // namespace stream_executor::rocm

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCPRIM_TARGET_GUARD_H_
