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
    name = "gemm_layernorm_gemm_cu",
    srcs = [
        "op.cu.cc",
    ],
    hdrs = [
        "op.h",
    ],
    deps = [":ck_deps",
            "@ck_archive//:ck"] +
    MATH_DEPS,
    gpu_copts = [
        "-std=c++17",
        "-fexceptions",
    ],
    nocopts = "-fno-exceptions",
)

tf_kernel_library(
    name = "gemm_layernorm_gemm",
    srcs = [
        "op.cc",
    ],
    hdrs = [
        "op.h",
    ],
    deps = [":ck_deps",
            "@ck_archive//:ck",
            ":gemm_layernorm_gemm_cu",
            "//tensorflow/core:new_nn_ops_op_lib",
    ] + MATH_DEPS,
    copts = ["-fexceptions",],
    nocopts = "-fno-exceptions",
)

tf_gpu_cc_test(
    name = "gemm_layernorm_gemm_test",
    size = "small",
    srcs = ["op_test.cc"],
    deps = [
        ":gemm_layernorm_gemm",
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