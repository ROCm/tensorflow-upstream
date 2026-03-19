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
# Run a single XLA test via RBE on MI250 (gfx90a) workers.
#
# Usage:
#   ./run_single_xla_rbe_test.sh <bazel_target> [test_filter]
#
# Examples:
#   ./run_single_xla_rbe_test.sh @@xla//xla/backends/profiler/gpu:rocm_tracer_test "RocmTracerTest.Counters"
#   ./run_single_xla_rbe_test.sh @@xla//xla/backends/profiler/gpu:rocm_tracer_test ""
#
# Exit code: 0 if the test passes, non-zero if it fails.

set -e
set -x

BAZEL_TARGET="$1"
TEST_FILTER="$2"

if [[ -z "$BAZEL_TARGET" ]]; then
    echo "Usage: $0 <bazel_target> [test_filter]"
    exit 1
fi

if [[ -z "${ROCM_PATH}" ]]; then
    ROCM_INSTALL_DIR=/opt/rocm/
else
    ROCM_INSTALL_DIR=$ROCM_PATH
fi
export ROCM_PATH=$ROCM_INSTALL_DIR
export TF_ROCM_RBE_DOCKER_IMAGE=rocm/tensorflow-build@sha256:a2191b80002ad851e5f7274994b0c62859874b38cc587115b116eff69b0948af

export PYTHON_BIN_PATH=$(which python3)
PYTHON_VERSION=$(python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export TF_PYTHON_VERSION=$PYTHON_VERSION
export TF_NEED_ROCM=1

if [ ! -d /tf ]; then
    mkdir -p /tf
fi

FILTER_ARG=""
if [[ -n "$TEST_FILTER" ]]; then
    FILTER_ARG="--test_filter=${TEST_FILTER}"
fi

bazel \
    --bazelrc=tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/rocm.bazelrc \
    --bazelrc=third_party/xla/build_tools/rocm/rocm_xla.bazelrc \
    test \
    --config=sigbuild_local_cache \
    --config=rocm \
    --config=xla_cpp_filters \
    --config=rocm_rbe \
    --repo_env=TF_ROCM_RBE_DOCKER_IMAGE=rocm/tensorflow-build@sha256:a2191b80002ad851e5f7274994b0c62859874b38cc587115b116eff69b0948af \
    --repo_env=TF_ROCM_RBE_POOL=linux_x64_gpu \
    --spawn_strategy=local \
    --jobs=$(nproc) \
    --test_output=all \
    --test_env=MIOPEN_FIND_ENFORCE=5 \
    --test_env=MIOPEN_FIND_MODE=1 \
    --action_env="ROCM_PATH=$ROCM_PATH" \
    --action_env=TF_ROCM_AMDGPU_TARGETS=gfx90a,gfx942 \
    --action_env=XLA_FLAGS=--xla_gpu_force_compilation_parallelism=16 \
    $FILTER_ARG \
    -- "$BAZEL_TARGET"
