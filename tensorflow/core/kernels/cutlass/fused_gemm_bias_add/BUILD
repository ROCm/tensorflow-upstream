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
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

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
    name = "cutlass_deps_fused_gemm_bias_add",
    hdrs = glob(["**/**/*.h"]), 
    visibility = ["//visibility:public"],
)


tf_kernel_library(
    name = "fused_gemm_bias_add",
    srcs = [
        "op.cc",
    ],
    gpu_srcs = [
        "op.cu.cc",
        "op.h",
        "cutlass_irrelevant.h",
        "api.h",
    ],
    hdrs = [
        "op.h",
        "api.h",
        "cutlass_irrelevant.h",
    ],
    deps = [":cutlass_deps_fused_gemm_bias_add"] +
    select({
        "//conditions:default": [
            "@cutlass_gpu//:headers",
        ],
    }) +
    MATH_DEPS + [
        "//tensorflow/core:new_nn_ops_op_lib",
        "//tensorflow/core/kernels/cutlass/include:cutlass_headers",
    ] + if_cuda([
        "//tensorflow/core/kernels:cuda_solvers",
    ])
)


tf_gpu_cc_test(
    name = "fused_gemm_bias_add_op_test",
    size = "small",
    srcs = ["fused_gemm_bias_add_op_test.cc"],
    deps = [
        ":fused_gemm_bias_add",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
