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
N_TEST_JOBS=4

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_BUILD__JOBS} concurrent test job(s)."
echo ""

# Run configure.
export PYTHON_BIN_PATH=`which python3`
PYTHON_VERSION=`python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
export TF_PYTHON_VERSION=$PYTHON_VERSION

if [ ! -d /tf ];then
    # The bazelrc files expect /tf to exist
    mkdir /tf
fi

#TODO weekly-sync 2026-09-15
EXCLUDED_TESTS=(
  # //tensorflow/c:c_api_experimental_test 
  CAPI_EXPERIMENTAL.LibraryNextPluggableDeviceLoadFunctions

  # //tensorflow/core/common_runtime:process_function_library_runtime_test_cpu
  ProcessFunctionLibraryRuntimeTest.MultiDevice_ResourceOutput_GPU
  ProcessFunctionLibraryRuntimeTest.MultiDevice_ErrorWhenBadTargetDevice

  # //tensorflow/core/common_runtime/pluggable_device:pluggable_device_plugin_init_test
  PluggableDevicePluginInitTest.StaticNPInitTest

  # //tensorflow/core/grappler/optimizers/data:split_utils_test
  SplitUtilsTest.MultiOutput

  # //tensorflow/core/kernels:matmul_op_test_cpu
  Test/FusedMatMulWithBiasOpTest/1.MatMul*
)

bazel --bazelrc=tensorflow/tools/tf_sig_build_dockerfiles/devel.usertools/cpu.bazelrc test \
          --config=sigbuild_local_cache \
          --config=pycpp \
          --config=rocm_cache
          --verbose_failures \
          --action_env=TF_NEED_ROCM=0 \
          --action_env=TF_PYTHON_VERSION=$PYTHON_VERSION \
          --local_test_jobs=${N_TEST_JOBS} \
          --repo_env=ROCM_PATH=/opt/rocm \
          --dynamic_mode=off \
          --test_timeout=400,600,1800,3600 \
          --test_env=TF_NUM_INTEROP_THREADS=4 \
          --test_env=TF_NUM_INTRAOP_THREADS=4 \
          --jobs=${N_BUILD_JOBS} \
          --test_filter=-$(IFS=: ; echo "${EXCLUDED_TESTS[*]}") \
          --test_env=HIP_VISIBLE_DEVICES=0 \
