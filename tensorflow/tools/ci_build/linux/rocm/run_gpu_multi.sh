#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
set -e
set -x

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
N_TEST_JOBS=1 # run tests serially

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s)."
echo ""

export TF_NEED_ROCM=1

if [ ! -d /tf ];then
    # The bazelrc files expect /tf to exist
    mkdir /tf
fi

# Failing tests:
# //tensorflow/python/distribute:multi_process_runner_test_2gpu TIMEOUT

bazel --bazelrc=tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/rocm.bazelrc test \
        --local_test_jobs=${N_TEST_JOBS} \
        --jobs=${N_BUILD_JOBS} \
        --local_ram_resources=60000 \
        --local_cpu_resources=15 \
        --verbose_failures \
        --config=rocm_ci_hermetic \
        --config=rocm_cache \
        --config=nonpip_multi_gpu \
        --config=sigbuild_local_cache \
        --repo_env="TF_ROCM_AMDGPU_TARGETS=$TARGET_ARCHS" \
        --repo_env=ROCM_DISTRO_URL="${ROCM_DISTRO_URL}" \
        --repo_env=ROCM_DISTRO_HASH="${ROCM_DISTRO_HASH}" \
        --repo_env=ROCM_PATH="" \
        --dynamic_mode=off
