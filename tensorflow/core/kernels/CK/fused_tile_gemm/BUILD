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
    name = "ck_deps_fused_tile_gemm",
    hdrs = glob(["**/**/*.h"]),
    visibility = ["//visibility:public"],
)

tf_kernel_library(
    name = "fused_tile_gemm",
    srcs = [
        "op.cc",
    ],
    gpu_srcs = [
        "op.cu.cc",
        "op.h",
        #"ck_irrelevant.h",
        "api.h",
    ],
    hdrs = [
        "op.h",
        "api.h",
        #"ck_irrelevant.h",
    ],
    deps = [":ck_deps_fused_tile_gemm"] +
    select({
        "//conditions:default": [
            "@ck_archive//:ck",
        ],
    }) +
    MATH_DEPS + [
        "//tensorflow/core:new_nn_ops_op_lib",
    ],
    copts = [
        "-std=c++17",
        "-fexceptions",
    ],
    nocopts = "-fno-exceptions",
)
