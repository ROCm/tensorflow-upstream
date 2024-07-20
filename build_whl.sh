bazel build -c opt --copt -g --strip=never --copt=-mavx --copt=-mavx2 --config=rocm --copt=-Wno-invalid-constexpr //tensorflow:libtensorflow_cc.so
bazel build -c opt --copt -g --strip=never --copt=-mavx --copt=-mavx2 --config=rocm  //tensorflow:libtensorflow_framework.so
