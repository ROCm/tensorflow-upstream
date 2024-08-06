#!/bin/sh
#****************************************************************#
# ScriptName: build.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2020-12-03 17:40
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2021-12-13 12:33
# Function: 
#***************************************************************#
set -e 

TF_PKG_LOC=/tmp/tensorflow_pkg
rm -f $TF_PKG_LOC/tensorflow*.whl

export USE_BAZEL_VERSION=0.26.1

yes "" | TF_NEED_CUDA=1 TF_CUDA_VERSION=11.7 CUDA_TOOLKIT_PATH=/usr/local/cuda-11.7 PYTHON_BIN_PATH=/usr/bin/python3 ./configure
pip3 uninstall -y tensorflow || true
bazel build -c opt --copt -g --strip=never --copt=-mavx --copt=-mavx2 --config=cuda \
        --copt -Wno-sign-compare \
        --copt -DCUB_NS_QUALIFIER=::cub \
        --action_env=TF_CUDA_COMPUTE_CAPABILITIES=7.0 \
        //tensorflow:libtensorflow_cc.so \
        //tensorflow:libtensorflow_framework.so
#bazel build --config=opt --config=rocm //tensorflow/tools/pip_package:build_pip_package --verbose_failures  &&
#bazel-bin/tensorflow/tools/pip_package/build_pip_package $TF_PKG_LOC

#echo y| pip uninstall tensorflow
#pip install $TF_PKG_LOC/tensorflow-1.15.5-cp37-cp37m-linux_x86_64.whl
