# Description:
# CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance
# matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE.txt"])

genrule(
    name = "config_h",
    srcs = [
        "include/ck/config.h.in",
    ],
    outs = [
        "include/ck/config.h",
    ],
    cmd = """
       awk '{gsub(/^#cmakedefine DTYPES \"@DTYPES@\"/, "/* #undef DTYPES*/");
             gsub(/^#cmakedefine CK_ENABLE_ALL_DTYPES @CK_ENABLE_ALL_DTYPES@/, "#define CK_ENABLE_ALL_DTYPES ON");
             gsub(/^#cmakedefine CK_ENABLE_INT8 @CK_ENABLE_INT8@/, "/* #undef CK_ENABLE_INT8*/");
             gsub(/^#cmakedefine CK_ENABLE_FP8 @CK_ENABLE_FP8@/, "/* #undef CK_ENABLE_FP8*/");
             gsub(/^#cmakedefine CK_ENABLE_BF8 @CK_ENABLE_BF8@/, "/* #undef CK_ENABLE_BF8*/");
             gsub(/^#cmakedefine CK_ENABLE_FP16 @CK_ENABLE_FP16@/, "/* #undef CK_ENABLE_FP16*/");
             gsub(/^#cmakedefine CK_ENABLE_BF16 @CK_ENABLE_BF16@/, "/* #undef CK_ENABLE_BF16*/");
             gsub(/^#cmakedefine CK_ENABLE_FP32 @CK_ENABLE_FP32@/, "/* #undef CK_ENABLE_FP32*/");
             gsub(/^#cmakedefine CK_ENABLE_FP64 @CK_ENABLE_FP64@/, "/* #undef CK_ENABLE_FP64*/");
             gsub(/^#cmakedefine CK_ENABLE_DL_KERNELS @CK_ENABLE_DL_KERNELS@/, "/* #undef CK_ENABLE_DL_KERNELS*/");
             gsub(/^#cmakedefine CK_ENABLE_INSTANCES_ONLY @CK_ENABLE_INSTANCES_ONLY@/, "/* #undef CK_ENABLE_INSTANCES_ONLY*/");
             gsub(/^#cmakedefine CK_USE_XDL @CK_USE_XDL@/, "#define CK_USE_XDL ON");
             gsub(/^#cmakedefine CK_USE_WMMA @CK_USE_WMMA@/, "/* #undef CK_USE_WMMA*/");
             gsub(/^#cmakedefine/, "//cmakedefine");print;}' $(<) > $(@)
    """,
)

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
        ":config_h",
    ],
    includes = [
        "include",
        "library/include/",
        #"library/include/ck/library/utility/",
        #"library/utility/",
        #".",
        #"include/ck",
    ],
    copts = [
        "-std=c++17",
        "-fexceptions",
    ],
)
