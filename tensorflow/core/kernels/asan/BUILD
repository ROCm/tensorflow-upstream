# Description:
#   quantization-specific OpKernels for hexagon

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gpu_cc_test",
    "tf_kernel_library",
    "tf_gpu_kernel_library",
)
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")


tf_gpu_kernel_library(
    name = "asan",
    srcs = [
        "asan.cu.cc",
    ],
    hdrs = ["asan.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",],
    # copts = ["-fsanitize=address"],
    # nocopts = "-fno-exceptions",
)


tf_gpu_cc_test(
    name = "asan_test",
    size = "small",
    srcs = ["test.cc"],
    deps = [
        ":asan",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)