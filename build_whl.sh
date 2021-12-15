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

bazel build --copt=-mavx --copt=-mavx2 --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_pkg

echo y| pip uninstall tensorflow
pip install  ~/tensorflow_pkg/tensorflow-1.15.5-cp37-cp37m-linux_x86_64.whl
