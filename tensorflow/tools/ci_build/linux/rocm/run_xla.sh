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

set -e
set -x

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
# If rocm-smi exists locally (it should) use it to find
# out how many GPUs we have to test with.
rocm-smi -i
STATUS=$?
if [ $STATUS -ne 0 ]; then TF_GPU_COUNT=1; else
   TF_GPU_COUNT=$(rocm-smi -i|grep 'Device ID' |grep 'GPU' |wc -l)
fi
TF_TESTS_PER_GPU=1
N_TEST_JOBS=$(expr ${TF_GPU_COUNT} \* ${TF_TESTS_PER_GPU})

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s)."
echo ""

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

export PYTHON_BIN_PATH=`which python3`
PYTHON_VERSION=`python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
export TF_PYTHON_VERSION=$PYTHON_VERSION
export TF_NEED_ROCM=1
export ROCM_PATH=$ROCM_INSTALL_DIR

if [ ! -d /tf ];then
        # The bazelrc files in /usertools expect /tf to exist
        mkdir /tf
    fi

# vvv TODO (rocm) weekly-sync-20260915 excluded tests
EXCLUDED_TESTS=(
    # @xla//xla/backends/gpu/autotuner:fission_backend_test_amdgpu_any
    CublasFissionBackendTest.CublasFallbackForBf16Bf16F32Algorithm

    # @xla//xla/backends/gpu/autotuner:triton_test_amdgpu_any
    TritonBackendTestSuite/TritonBackendTest.CostModelOptions_*

    # @xla//xla/backends/gpu/transforms:gemm_rewriter_test_amdgpu_any
    GemmRewriteTest.CheckCustomCallHipblasLtBF16
    ParameterizedGemmRewriteTest.GemmTypeCombinationCheck

    # @xla//xla/service/gpu:dot_algorithm_support_test_amdgpu_any
    F8E5M2Tests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_*
    F8E4M3FNTests/DotAlgorithmSupportTest.AlgorithmIsSupportedFromCudaCapability/dot_any_f8_any_f8_f32_*

    # @xla//xla/backends/gpu/codegen/triton/dot_algorithms-test_amdgpu_any
    BlasAlgorithmTest.Algorithm_BF16_BF16_F32*

    # @xla//xla/service/gpu:float_support_test_amdgpu_any
    FloatSupportTestWithCublas.MixedTypeDotIsNotUpcasted

    # @xla//xla/tests:convolution_1d_autotune_disabled_test_amdgpu_any
    Convolve1D1WindowTest_Instantiation/Convolve1D1WindowTestHalf.Convolve1D1Window/6

    # @xla//xla/tests:convolution_autotune_disabled_test_amdgpu_any
    ConvolutionTest.Convolve3D_1x4x2x3x3_2x2x2x3x3_Valid

    # @xla//xla/backends/gpu/profiler:kernel_name_tracer_test
    KernelNameTracerTest.Create
    KernelNameTracerTest.CaptureKernelNames
    KernelNameTracerTest.CaptureKernelNamesFromCommandBufferThunk
)

bazel --bazelrc=tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/rocm.bazelrc test \
    --config=sigbuild_local_cache \
    --config=rocm \
    --config=xla_cpp_filters \
    --test_output=errors \
    --local_test_jobs=${N_TEST_JOBS} \
    --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
    --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
    --test_env=MIOPEN_FIND_ENFORCE=5 \
    --test_env=MIOPEN_FIND_MODE=1 \
    --action_env="ROCM_PATH=$ROCM_PATH" \
    --action_env=XLA_FLAGS=--xla_gpu_force_compilation_parallelism=16 \
    --dynamic_mode=off \
    --test_filter=-$(IFS=: ; echo "${EXCLUDED_TESTS[*]}") \
    -- @xla//xla/... \
    # ^^^ TODO (rocm) weekly-sync-20251021 excluded test files
