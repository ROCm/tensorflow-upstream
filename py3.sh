#!/usr/bin/env bash

export ASAN_OPTIONS="detect_leaks=0,detect_odr_violation=0,allocator_may_return_null=1"
export LD_PRELOAD="/opt/rocm-6.3.0.1/lib/llvm/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so /opt/rocm-6.3.0.1/lib/asan/libhsa-runtime64.so"
export LD_LIBRARY_PATH="/opt/rocm-6.3.0.1/lib/asan:$LD_LIBRARY_PATH"
export HSA_XNACK=1

exec bash -c "/usr/bin/python3 ${*@Q}"
