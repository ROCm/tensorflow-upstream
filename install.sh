#export TEST_TMPDIR=
export TF_PLATFORM=default
export TF_NEED_RTP=1
export TF_DEVICE=gpu
export PATH=/usr/local/bin:/usr/lib/jvm/jre-1.8.0/bin:$PATH
export JAVA_HOME=/usr/lib/jvm/jre-1.8.0/

export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export PYTHON_BIN_PATH=$(which python)
export USE_DEFAULT_PYTHON_LIB_PATH=1
export CC_OPT_FLAGS="-march=native"

export CUDA_TOOLKIT_PATH=/usr/local/cuda/
export TF_CUDA_VERSION=10.1
export TF_CUDALIB_VERSION=10

export CUDNN_INSTALL_PATH=/usr/local/cuda/
export TF_CUDNN_VERSION=7

export NCCL_INSTALL_PATH=/usr/local/cuda/
export TF_NCCL_VERSION=2.3.7

export TF_CUDA_CLANG=0
export TF_CUDA_COMPUTE_CAPABILITIES="6.0,6.1,7.0,7.5"

export TF_NEED_CUDA=1
export TF_NEED_IGNITE=0
export TF_NEED_PANGU=0
export TF_NEED_PANGU_TEMP=0
export TF_NEED_PAI=0
export TF_NEED_BRPC=0
export TF_NEED_STAR=0
export TF_NEED_JEMALLOC=0
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
export TF_NEED_S3=0
export TF_NEED_GDR=0
export TF_NEED_VERBS=0
export TF_NEED_OPENCL=0
export TF_NEED_PAI_TRT=0
export TF_NEED_MPI=0
export TF_ENABLE_XLA=1
export TF_NEED_ZOOKEEPER=0
export TF_NEED_FPGA=1
export TF_NEED_PAI_ALIFPGA=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_MKL=0
export TF_NEED_PORSCHE=0
export TF_NEED_MONOLITH=1
export TF_NEED_METRIC=0

python ./configure.py
declare -a targets=("//tensorflow/cc:cc_op_gen_main"
                    "//tensorflow/core:op_gen_lib"
#                     "//tensorflow/core:op_gen_overrides_proto_cc"
                    "//tensorflow:libtensorflow_cc.so"
                    "//tensorflow/core:test"
                    "//tensorflow/core:testlib"
                    "//tensorflow/core/kernels:ops_testutil"
                    "//tensorflow:libnew_nn_ops.so")
declare -a install_targets=("cc/libcc_op_gen_main.a"
                            "core/libop_gen_lib.a"
#                            "core/libop_gen_overrides_proto_cc.a"
                            "libtensorflow_cc.so"
                            "core/libtest.so"
                            "core/libtestlib.so"
                            "core/kernels/libops_testutil.so"
                            "libnew_nn_ops.so")
## now loop through the above array
for target in "${targets[@]}"
do
    bazel build --define framework_shared_object=false --config=cuda -c opt --copt -g --copt -mavx2 --copt -mfma --copt -DRTP_PLATFORM --copt -D_GLIBCXX_USE_CXX11_ABI=0 --copt -DGOOGLE_CUDA=1 --copt -fno-canonical-system-headers $target
done

# python
# bazel build --define framework_shared_object=false -c opt --copt -g //tensorflow/tools/pip_package:build_pip_package
# ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg/
# PYTHON_INSTALL_ROOT=/home/liukan.lk/tensorflow_install
# pip install --upgrade /tmp/tensorflow_pkg/* --target=$PYTHON_INSTALL_ROOT
# touch $PYTHON_INSTALL_ROOT/google/__init__.py

# zipfile.LargeZipFile: Filesize would require ZIP64 extensions:
# https://github.com/tensorflow/tensorflow/issues/5538

# test
# bazel test --config=cuda --test_tag_filters=-no_oss,-oss_serial,-no_gpu,-benchmark-test -k \
#     --test_lang_filters=cc --jobs=96 --test_timeout 300 \
#     --build_tests_only --test_output=errors --local_test_jobs=8 \
#     --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute -- \
#     //tensorflow/... -//tensorflow/compiler/... -//tensorflow/contrib/...

# install
EXTERNAL_DIR="../_external"
EXTERNAL_DIR=`readlink -f $EXTERNAL_DIR`
HEADER_DIR=$EXTERNAL_DIR"/usr/local/include/"

CURRENT_DIR=`basename $PWD`
BAZEL_CACHE_DIR=`readlink bazel-$CURRENT_DIR`/../../
BAZEL_EXTERNAL_DIR=$BAZEL_CACHE_DIR"/external/"

for target in "${install_targets[@]}"
do
    IFS='/' read -ra path <<< "$target"
    rm $EXTERNAL_DIR/usr/local/lib64/${path[-1]} -f
    cp bazel-bin/tensorflow/$target $EXTERNAL_DIR/usr/local/lib64/
done

# copy header
rm $HEADER_DIR/tensorflow -rf
mkdir -p $HEADER_DIR/tensorflow
find tensorflow/core -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
find tensorflow/core -name '*.proto' -exec cp --parents \{\} $HEADER_DIR/ \;
find tensorflow/c -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
find tensorflow/cc -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
if [ -d bazel-out/local-opt ]; then
    cd bazel-out/local-opt/genfiles
    find tensorflow/ -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
    cd -
fi
if [ -d bazel-out/local_linux-opt ]; then
   cd bazel-out/local_linux-opt/genfiles
   find tensorflow/ -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
   cd -
fi
if [ -d bazel-out/k8-opt ]; then
   cd bazel-out/k8-opt/genfiles
   find tensorflow/ -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
   cd -
fi

EIGEN_DIR=$BAZEL_EXTERNAL_DIR"eigen_archive"
rm $HEADER_DIR/eigen3 -rf
cp -r $EIGEN_DIR $HEADER_DIR/eigen3
mkdir -p $HEADER_DIR/third_party/
cp -r third_party/eigen3/ $HEADER_DIR/third_party/

NSYNC_DIR=$BAZEL_EXTERNAL_DIR"nsync/public/"
cd $NSYNC_DIR
find . -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
cd -

ABSL_DIR=$BAZEL_EXTERNAL_DIR"com_google_absl/absl/"
rm $HEADER_DIR/absl -rf
cp -r $ABSL_DIR $HEADER_DIR/absl

DITING_DIR=$BAZEL_EXTERNAL_DIR"diting_repo/sdk/include/diting/"
rm $HEADER_DIR/diting -rf
cp -r $DITING_DIR $HEADER_DIR/diting


FARMHASH=$BAZEL_EXTERNAL_DIR"farmhash_archive/src/farmhash.h"
cp $FARMHASH $HEADER_DIR
