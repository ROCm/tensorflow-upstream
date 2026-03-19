#!/usr/bin/env bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
#
# Runs XLA unit tests on ROCm platform for MI250 (gfx90a) via Remote Build
# Execution (RBE) using the EngFlow cluster (wardite.cluster.engflow.com).
#
# Differences from run_xla.sh:
#   - Loads rocm_xla.bazelrc (provides --config=rocm_rbe with EngFlow settings)
#   - Adds --config=rocm_rbe  (remote executor, BES streaming, --jobs=200)
#   - Removes --local_test_jobs  (parallelism is managed by RBE, not locally)
#   - Removes --test_env=TF_GPU_COUNT/TF_TESTS_PER_GPU  (not applicable to RBE workers)
#   - Builds locally (--spawn_strategy=local, --jobs=nproc) for speed; only
#     test execution runs via RBE
#   - Sets TF_ROCM_RBE_DOCKER_IMAGE (rocm/tensorflow-build:latest-noble-python3.11-rocm7.1.1)
#     to match the local build environment so remote workers use the same
#     glibc/libstdc++ as the locally-built binaries
#
# Prerequisites:
#   - TLS client certificate and key at /tf/certificates/ci-cert.{crt,key}

set -e
set -x

# First positional argument (if any) specifies the ROCM_INSTALL_DIR
if [[ -n $1 ]]; then
    ROCM_INSTALL_DIR=$1
else
    if [[ -z "${ROCM_PATH}" ]]; then
        ROCM_INSTALL_DIR=/opt/rocm/
    else
        ROCM_INSTALL_DIR=$ROCM_PATH
    fi
fi

TF_ROCM_AMDGPU_TARGETS=gfx90a,gfx942

N_BUILD_JOBS=$(nproc)

export PYTHON_BIN_PATH=`which python3`
PYTHON_VERSION=`python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
export TF_PYTHON_VERSION=$PYTHON_VERSION
export TF_NEED_ROCM=1
export ROCM_PATH=$ROCM_INSTALL_DIR
export TF_ROCM_RBE_DOCKER_IMAGE=rocm/tensorflow-build@sha256:a2191b80002ad851e5f7274994b0c62859874b38cc587115b116eff69b0948af

if [ ! -d /tf ]; then
    # The bazelrc files in /usertools expect /tf to exist
    mkdir /tf
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
    --jobs=${N_BUILD_JOBS} \
    --test_output=errors \
    --test_env=MIOPEN_FIND_ENFORCE=5 \
    --test_env=MIOPEN_FIND_MODE=1 \
    --action_env="ROCM_PATH=$ROCM_PATH" \
    --action_env=TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS}" \
    --action_env=XLA_FLAGS=--xla_gpu_force_compilation_parallelism=16 \
    -- @xla//xla/...
