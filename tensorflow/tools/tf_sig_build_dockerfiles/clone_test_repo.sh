#!/bin/bash
# Copyright 2026 The TensorFlow Authors
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


_DL_FLD=${TF_TESTING_FL:-'/'}
cd "${_DL_FLD}" || exit 2


clone_benchmark() {
    local benchmarks_repo='https://github.com/tensorflow/benchmarks'
    local tf_ver
    tf_ver=$(grep -oP "_VERSION = '\K[0-9]+\.[0-9]+" \
        "${_DL_FLD}/tensorflow/tensorflow/tools/pip_package/setup.py.tpl")
    if [ -z "${tf_ver}" ]; then
        echo "TF version cannot be found..."
        exit 1
    else
        echo "tf_ver=${tf_ver}"
    fi
    # For TF version higher than 2.12, use the new AMD benchmarks repo
    if (( ${tf_ver//./} > 212 ));then
        benchmarks_repo='https://github.com/ROCmSoftwarePlatform/benchmarks'
        benchmark_branch='-b tf2.13-compatible'
    fi
    git clone ${benchmark_branch} ${benchmarks_repo}
}

git clone https://github.com/tensorflow/models.git
git clone https://github.com/tensorflow/examples.git
git clone https://github.com/tensorflow/autograph.git
clone_benchmark
