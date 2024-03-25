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
rocm-smi -i
STATUS=$?
if [ $STATUS -ne 0 ]; then TF_GPU_COUNT=1; else
   TF_GPU_COUNT=$(rocm-smi -i|grep 'ID' |grep 'GPU' |wc -l)
fi
HIP_VISIBLE_DEVICES=1
TF_GPU_COUNT=1
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
        ROCM_INSTALL_DIR=/opt/rocm-6.0.2
    else
        ROCM_INSTALL_DIR=$ROCM_PATH
    fi
fi

# Run configure.
export PYTHON_BIN_PATH=`which python3`

PYTHON_VERSION=`python3 -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"`
export TF_PYTHON_VERSION=$PYTHON_VERSION
export TF_NEED_ROCM=1
export ROCM_PATH=$ROCM_INSTALL_DIR

if [ -f /usertools/rocm.bazelrc ]; then
	# Use the bazelrc files in /usertools if available
 	if [ ! -d /tf ];then
           # The bazelrc files in /usertools expect /tf to exist
           mkdir /tf
        fi
	bazel \
	     --bazelrc=/usertools/rocm.bazelrc \
             test \
	     --jobs=30 \
	     --local_ram_resources=60000 \
	     --local_cpu_resources=15 \
	     --local_test_jobs=${N_TEST_JOBS} \
             --config=sigbuild_local_cache \
             --config=rocm \
             --config=pycpp \
             --action_env=OPENBLAS_CORETYPE=Haswell \
             --action_env=TF_PYTHON_VERSION=$PYTHON_VERSION \
             --action_env=TF_ENABLE_ONEDNN_OPTS=0 \
 			 --test_env=TF_NUM_INTEROP_THREADS=16 \
			 --test_env=TF_NUM_INTRAOP_THREADS=16 \
			 --test_env=XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16" \
			 --test_env=HIP_VISIBLE_DEVICES=1 \
             --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
             --test_env=TF_GPU_COUNT=$TF_GPU_COUNT
			  @local_xla//xla/tests/...
else
	# Legacy style: run configure then build
	yes "" | $PYTHON_BIN_PATH configure.py

	# Run bazel test command. Double test timeouts to avoid flakes.
	bazel test \
	      --config=rocm \
	      -k \
	      --test_tag_filters=gpu,-no_oss,-oss_excluded,-oss_serial,-no_gpu,-no_rocm,-benchmark-test,-rocm_multi_gpu,-tpu,-v1only \
	      --jobs=30 \
	      --local_ram_resources=60000 \
	      --local_cpu_resources=15 \
	      --local_test_jobs=${N_TEST_JOBS} \
	      --test_env=TF_GPU_COUNT=$TF_GPU_COUNT \
	      --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU \
	      --test_env=HSA_TOOLS_LIB=libroctracer64.so \
	      --test_env=TF_PYTHON_VERSION=$PYTHON_VERSION \
		 --test_env=TF_NUM_INTEROP_THREADS=16 \
 		 --test_env=TF_NUM_INTRAOP_THREADS=16 \
		 --test_env=XLA_FLAGS="--xla_gpu_force_compilation_parallelism=16" \
		 --test_env=HIP_VISIBLE_DEVICES=1 \
	      --action_env=OPENBLAS_CORETYPE=Haswell \
        --action_env=TF_ENABLE_ONEDNN_OPTS=0 \
	      --test_timeout 920,2400,7200,9600 \
	      --build_tests_only \
	      --test_output=errors \
	      --test_sharding_strategy=disabled \
	      --test_size_filters=small,medium,large \
	      --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute \
	      -- \
	       @local_xla//xla/tests/...
fi
