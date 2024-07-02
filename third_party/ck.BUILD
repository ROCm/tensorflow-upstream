# Description:
# CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance
# matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE.txt"])

filegroup(
    name = "ck_header_files",
    srcs = glob([
        "include/**",
    ]),
)

filegroup(
    name = "ck_util_header_files",
    srcs = glob([
        "library/include/ck/library/**",
    ]),
)

cc_library(
    name = "ck",
    hdrs = [
        ":ck_header_files",
        ":ck_util_header_files",
    ],
    includes = [
        "include",
        "library/include/ck/library",
    ],
)
