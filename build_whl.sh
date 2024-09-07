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

# First positional argument (if any) specifies the ROCM_INSTALL_DIR
ROCM_INSTALL_DIR=`readlink -f /opt/rocm`
if [ "$#" -ge 1 ]; then
    ROCM_INSTALL_DIR=$1
fi
export ROCM_TOOLKIT_PATH=$ROCM_INSTALL_DIR
export USE_BAZEL_VERSION=0.26.1

#--explain=explanation.log --verbose_explanations \


#yes "" | TF_NEED_ROCM=1 ROCM_TOOLKIT_PATH=${ROCM_INSTALL_DIR} PYTHON_BIN_PATH=/usr/bin/python3 PYTHON_BIN_PATH=/usr/bin/python3 ./configure
#pip3 uninstall -y tensorflow || true
bazel build -c opt --copt -g --strip=never --copt=-mavx --copt=-mavx2 --config=rocm \
            --copt -Wno-invalid-constexpr \
            --copt -Wno-deprecated-builtins \
            --copt -Wno-unused-command-line-argument \
            --copt -Wno-unused-but-set-variable \
            --copt -Wno-final-dtor-non-final-class \
            --copt -Wno-deprecated-declarations \
            --copt -Wno-sign-compare \
            --copt -Wno-bitwise-instead-of-logical \
            --copt=-Wno-error=array-parameter  \
            --copt=-Wno-error=array-bounds \
            --cxxopt -std=c++17    \
            --action_env=TF_ROCM_AMDGPU_TARGETS=gfx942 -s  \
            //tensorflow:libtensorflow_cc.so \
            //tensorflow:libtensorflow_framework.so

#bazel build --config=opt --config=rocm //tensorflow/tools/pip_package:build_pip_package --verbose_failures  &&
#bazel-bin/tensorflow/tools/pip_package/build_pip_package $TF_PKG_LOC

#echo y| pip uninstall tensorflow
#pip install  ~/tensorflow_pkg/tensorflow-1.15.5-cp37-cp37m-linux_x86_64.whl
