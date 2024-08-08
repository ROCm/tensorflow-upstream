# Description:
#   quantization-specific OpKernels for hexagon

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gpu_cc_test",
    "tf_gpu_kernel_library",
    "tf_kernel_library",
)
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")

MATH_DEPS = [
    "//tensorflow/core/kernels:bounds_check",
    "//tensorflow/core/kernels:fill_functor",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:math_grad",
    "//tensorflow/core:math_ops_op_lib",
    "//third_party/eigen3",
]

cc_library(
    name = "ck_deps",
    hdrs = glob(["**/**/*.h"]),
    visibility = ["//visibility:public"],
)

tf_gpu_kernel_library(
    name = "hip_gpu_layer_norm_cu",
    srcs = [
        "hip_gpu_layer_norm_op_gpu_layer_norm.cu.cc",
    ],
    hdrs = [
        "hip_gpu_layer_norm_op_gpu_layer_norm.h",
    ],
    deps = MATH_DEPS,
    gpu_copts = [
        "-std=c++17",
        "-fexceptions",
    ],
    nocopts = "-fno-exceptions",
)

tf_kernel_library(
    name = "hip_gpu_layer_norm",
    srcs = [
        "hip_gpu_layer_norm_op_gpu_layer_norm.cc",
    ],
    hdrs = [
        "hip_gpu_layer_norm_op_gpu_layer_norm.h",
    ],
    deps = MATH_DEPS,
    copts = ["-fexceptions",],
    nocopts = "-fno-exceptions",
)

tf_gpu_cc_test(
    name = "hip_gpu_layer_norm_test",
    size = "small",
    srcs = ["gpu_layer_norm_op_test.cc"],
    deps = [
        ":hip_gpu_layer_norm",
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