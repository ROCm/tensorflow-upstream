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
# If rocm-smi exists locally (it should) use it to find
# out how many GPUs we have to test with.
TF_GPU_COUNT=4
TF_TESTS_PER_GPU=1
N_TEST_JOBS=$(expr ${TF_GPU_COUNT} \* ${TF_TESTS_PER_GPU})

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s)."
echo ""

export TF_NEED_ROCM=1


TARGET_ARCHS="gfx90a"
ROCM_DISTRO_URL="https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-10.0.0.tar.gz"
ROCM_DISTRO_HASH="1c5e807875d26a2470ecc7323daa5b5b9009208a55c3290ac255a909cde15fc6"

if [ ! -d /tf ];then
    # The bazelrc files expect /tf to exist
        mkdir /tf
fi

# vvv TODO (rocm) weekly-sync-20251224 excluded tests
EXCLUDED_TESTS=(
    #  //tensorflow/core/grappler/optimizers:auto_mixed_precision_test_gpu
    AutoMixedPrecisionGfx1103Test.SupportsFp16

    # //tensorflow/core/kernels:matmul_op_test_gpu
    Test/FusedMatMulWithBiasOpTest/1.MatMul*

    # //tensorflow/core/common_runtime:process_function_library_runtime_test_gpu
    ProcessFunctionLibraryRuntimeTest.MultiDevice_ResourceOutput_GPU

    # //tensorflow/core/util/autotune_maps:autotune_serialize_test_gpu
    AutotuneSerializeTest.Consistency
    AutotuneSerializeTest.VersionControl

    # //tensorflow/core/profiler/backends/gpu:device_tracer_test
    DeviceTracerTest.StartTwoTracers
    DeviceTracerTest.TraceToXSpace
)

# Run bazel test command. Double test timeouts to avoid flakes.
bazel --bazelrc=tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/rocm.bazelrc test \
    --config=rocm_ci_hermetic \
    --config=rocm_cache \
    --config=sigbuild_local_cache \
    --config=pycpp \
    --jobs=${N_BUILD_JOBS} \
    --local_test_jobs=${N_TEST_JOBS} \
    --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
    --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
    --test_env=MIOPEN_DEBUG_CONV_WINOGRAD=0 \
    --repo_env="TF_ROCM_AMDGPU_TARGETS=$TARGET_ARCHS" \
    --repo_env=ROCM_DISTRO_URL="${ROCM_DISTRO_URL}" \
    --repo_env=ROCM_DISTRO_HASH="${ROCM_DISTRO_HASH}" \
    --repo_env=ROCM_PATH="" \
    --build_tests_only \
    --test_output=errors \
    --verbose_failures \
    --test_sharding_strategy=disabled \
    --dynamic_mode=off \
    --test_filter=-$(IFS=: ; echo "${EXCLUDED_TESTS[*]}") \
    --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute
