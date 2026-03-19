#!/usr/bin/env bash
# ==============================================================================
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Run a single GPU (pycpp) test by bazel target and test filter.
#
# Usage:
#   ./run_single_gpu_test.sh <bazel_target> <test_filter>
#
# Examples:
#   ./run_single_gpu_test.sh //tensorflow/core/kernels:matmul_op_test_gpu "Test/FusedMatMulWithBiasOpTest/1.MatMul*"
#   ./run_single_gpu_test.sh //tensorflow/core/profiler/backends/gpu:device_tracer_test "DeviceTracerTest.TraceToXSpace"
#
# Exit code: 0 if the test passes, non-zero if it fails.

set -e
set -x

BAZEL_TARGET="$1"
TEST_FILTER="$2"

if [[ -z "$BAZEL_TARGET" || -z "$TEST_FILTER" ]]; then
    echo "Usage: $0 <bazel_target> <test_filter>"
    echo "  bazel_target: e.g. //tensorflow/core/kernels:matmul_op_test_gpu"
    echo "  test_filter:  e.g. DeviceTracerTest.TraceToXSpace"
    exit 1
fi

# GPU detection
N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
rocm-smi -i > /dev/null 2>&1
STATUS=$?
if [ $STATUS -ne 0 ]; then TF_GPU_COUNT=1; else
    TF_GPU_COUNT=$(rocm-smi -i | grep 'Device ID' | grep 'GPU' | wc -l)
fi
TF_TESTS_PER_GPU=1
N_TEST_JOBS=$(expr ${TF_GPU_COUNT} \* ${TF_TESTS_PER_GPU})

# ROCM path
if [[ -z "${ROCM_PATH}" ]]; then
    ROCM_INSTALL_DIR=/opt/rocm/
else
    ROCM_INSTALL_DIR=$ROCM_PATH
fi
export ROCM_PATH=$ROCM_INSTALL_DIR

export PYTHON_BIN_PATH=$(which python3)
PYTHON_VERSION=$(python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export TF_PYTHON_VERSION=$PYTHON_VERSION
export TF_NEED_ROCM=1

# Detect GPU architecture
TARGET_ARCHS=$(rocminfo | grep "Name: *gfx" | awk '/Name:/ {print $2}' | sort -u)
if [ -z "$TARGET_ARCHS" ]; then
    echo "No gpu found"
    exit 1
fi

if [ ! -d /tf ]; then
    mkdir -p /tf
fi

yes "" | $PYTHON_BIN_PATH configure.py

bazel --bazelrc=tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/rocm.bazelrc test \
    --config=rocm \
    --config=sigbuild_local_cache \
    --config=pycpp \
    -k \
    --jobs=${N_BUILD_JOBS} \
    --local_test_jobs=${N_TEST_JOBS} \
    --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
    --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
    --test_env=MIOPEN_DEBUG_CONV_WINOGRAD=0 \
    --repo_env="TF_ROCM_AMDGPU_TARGETS=$TARGET_ARCHS" \
    --repo_env="ROCM_PATH=$ROCM_PATH" \
    --build_tests_only \
    --test_output=all \
    --verbose_failures \
    --test_sharding_strategy=disabled \
    --test_filter="$TEST_FILTER" \
    --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
    -- "$BAZEL_TARGET"
