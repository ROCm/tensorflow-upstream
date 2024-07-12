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
    name = "gather_attention_cu",
    srcs = [
        "op.cu.cc",
        "op.h",
    ] + glob(["*.hpp"]) + glob(["instances/*.cpp"]),
    hdrs = ["op.h"] + glob(["*.hpp"]),
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
    name = "gather_attention",
    srcs = [
        "op.cc",
    ],
    hdrs = ["op.h"] + glob(["*.hpp"]),
    deps = [":ck_deps",
            "@ck_archive//:ck",
            ":gather_attention_cu",
            "//tensorflow/core:new_nn_ops_op_lib",
    ] + MATH_DEPS,
    copts = ["-fexceptions",],
    nocopts = "-fno-exceptions",
)


tf_gpu_cc_test(
    name = "gather_attention_test",
    size = "small",
    srcs = ["op_test.cc"],
    deps = [
        ":gather_attention",
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