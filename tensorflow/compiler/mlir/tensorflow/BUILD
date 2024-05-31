load("@local_config_mlir//:tblgen.bzl", "gentbl", "gentbl_cc_library", "td_library")
load("@local_config_mlir//:mlir_configure.bzl", "is2019")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_native_cc_binary")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["@local_config_mlir//:subpackages"],
    packages = [
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/google/xla/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/python/...",
    ],
)

filegroup(
    name = "tensorflow_ops_td_files",
    srcs = [
        "ir/tf_generated_ops.td",
        "ir/tf_op_interfaces.td",
        "ir/tf_op_base.td",
        "ir/tf_ops.td",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)

gentbl(
    name = "tensorflow_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_ops.cc.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/tf_ops.md",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "tensorflow_executor_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_executor.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_executor.cc.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/tf_executor.md",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/tf_executor_ops.td",
    td_srcs = [
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
    ] + (
    ["@local_config_mlir//:include/mlir/Dialect/StandardOps/Ops.td"]
    if is2019()
    else [
        "@local_config_mlir//:include/mlir/Dialect/Transform/IR/TransformOps.td",
        "@local_config_mlir//:include/mlir/Dialect/Arith/IR/ArithOps.td",
    ]),
)

gentbl(
    name = "tensorflow_device_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_device.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_device.cc.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/tf_device.md",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/tf_device_ops.td",
    td_srcs = [
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
    ] + (
    ["@local_config_mlir//:include/mlir/Dialect/StandardOps/Ops.td"]
    if is2019()
    else [
        "@local_config_mlir//:include/mlir/Dialect/Transform/IR/TransformOps.td",
        "@local_config_mlir//:include/mlir/Dialect/Arith/IR/ArithOps.td",
    ]),
)

gentbl(
    name = "tensorflow_canonicalize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_canonicalize.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/canonicalize.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

cc_library(
    name = "tensorflow",
    srcs = [
        "ir/control_flow_ops.cc",
        "ir/tf_device.cc",
        "ir/tf_executor.cc",
        "ir/tf_executor.cc.inc",
        "ir/tf_executor.h.inc",
        "ir/tf_ops.cc",
        "ir/tf_ops.cc.inc",
        "ir/tf_ops.h.inc",
        "transforms/cluster_formation.cc",
        "transforms/cluster_outlining.cc",
        "transforms/executor_island_coarsening.cc",
        "transforms/functional_control_flow_to_cfg.cc",
        "transforms/generated_canonicalize.inc",
        "transforms/generated_optimize.inc",
        "transforms/graph_pruning.cc",
        "transforms/optimize.cc",
        "transforms/raise_control_flow.cc",
        "translate/control_to_executor_dialect.cc",
        "translate/executor_to_control_dialect.cc",
        "utils/verification_utils.cc",
    ],
    hdrs = [
        "ir/control_flow_ops.h",
        "ir/tf_device.h",
        "ir/tf_executor.h",
        "ir/tf_ops.h",
        "ir/tf_traits.h",
        "ir/tf_types.def",
        "ir/tf_types.h",
        "ir/tf_side_effects.h",
        "transforms/passes.h",
        "utils/attribute_utils.h",
        "transforms/side_effect_analysis.h",
        "transforms/per_function_aggregate_analysis.h",
        "transforms/resource_alias_analysis.h",
        "utils/cluster_util.h",
        "utils/verification_utils.h",
      "ir/tf_op_interfaces.h",
      "ir/tf_structs.h",
      "ir/tf_verifiers.h",
    ],
    copts = ["-std=c++17"],
    includes = ["include"],
    deps = [
        ":tensorflow_canonicalize_inc_gen",
        ":tensorflow_device_ops_inc_gen",
        ":tensorflow_executor_inc_gen",
        ":tensorflow_ops_inc_gen",
        ":tensorflow_optimize_inc_gen",
        "//tensorflow/compiler/mlir/lite:validators",
        "//tensorflow/core:lib",
#        "//tensorflow/core:util",
        "//tensorflow/core:framework",
        "//tensorflow/compiler/tf2xla:tf2xla_defs",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:Dialect",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:DestinationStyleOpInterfaceIncGen",
        "@local_config_mlir//:ParallelCombiningOpInterfaceIncGen",
        "@local_config_mlir//:ShapedOpInterfacesIncGen",
        "@local_config_mlir//:TilingInterfaceIncGen",
        "@local_config_mlir//:TensorOpsIncGen",
        ":TypesIncGen",
        ":AttributesIncGen",
        ":DialectIncGen",
        ":tf_pass_inc_gen",
        ":tensorflow_op_interfaces_inc_gen",
    ],
    # TODO(jpienaar): Merge in the dialect registration.
    alwayslink = 1,
)

# Library with TensorFlow dialect static initialization.
cc_library(
    name = "tensorflow_dialect_registration",
    srcs = ["ir/dialect_registration.cc"],
    copts = ["-std=c++17"],
    deps = [
        ":tensorflow",
        "@local_config_mlir//:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "convert_graphdef",
    srcs = [
        "translate/export_graphdef.cc",
        "translate/import_model.cc",
    ],
    hdrs = [
        "translate/export_graphdef.h",
        "translate/import_model.h",
    ],
    copts = ["-std=c++17"],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":export_tf_dialect_op",
        ":export_utils",
        ":mangling_util",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        "//tensorflow/compiler/jit:shape_inference_helpers",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:Support",
    ] + (["@local_config_mlir//:StandardOps"] if is2019() else []),
)

cc_library(
    name = "import_utils",
    srcs = [
        "utils/import_utils.cc",
    ],
    hdrs = [
        "utils/import_utils.h",
    ],
    copts = ["-std=c++17"],
    deps = [
        ":error_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
    ],
)

cc_library(
    name = "export_utils",
    srcs = [
        "utils/export_utils.cc",
    ],
    hdrs = [
        "utils/export_utils.h",
    ],
    copts = ["-std=c++17"],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":mangling_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:Support",
    ] + (["@local_config_mlir//:StandardOps"] if is2019() else []),
)

cc_library(
    name = "export_tf_dialect_op",
    srcs = [
        "translate/derived_attr_populator.inc",
        "translate/export_tf_dialect_op.cc",
    ],
    hdrs = [
        "translate/export_tf_dialect_op.h",
    ],
    copts = ["-std=c++17"],
    deps = [
        ":convert_type",
        ":export_utils",
        ":tensorflow",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "translate_tf_dialect_op",
    srcs = ["translate/translate_tf_dialect_op.cc"],
    copts = ["-std=c++17"],
    deps = [
        ":export_tf_dialect_op",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Translation",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_pass",
    srcs = ["translate/mlir_roundtrip_pass.cc"],
    hdrs = ["translate/mlir_roundtrip_pass.h"],
    copts = ["-std=c++17"],
    deps = [
        ":convert_graphdef",
        ":mlir_roundtrip_flags",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
    ] + (["@local_config_mlir//:StandardOps"] if is2019() else []),
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_flags",
    srcs = ["translate/mlir_roundtrip_flags.cc"],
    hdrs = ["translate/mlir_roundtrip_flags.h"],
    copts = ["-std=c++17"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
    ],
)

cc_library(
    name = "convert_type",
    srcs = ["utils/convert_type.cc"],
    hdrs = ["utils/convert_type.h"],
    copts = ["-std=c++17"],
    deps = [
        ":tensorflow",
        ":tensorflow_dialect_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Support",
    ],
)

cc_library(
    name = "convert_tensor",
    srcs = ["utils/convert_tensor.cc"],
    hdrs = ["utils/convert_tensor.h"],
    copts = ["-std=c++17"],
    deps = [
        ":convert_type",
        ":mangling_util",
        ":tensorflow",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "mangling_util",
    srcs = ["utils/mangling_util.cc"],
    hdrs = ["utils/mangling_util.h"],
    copts = ["-std=c++17"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "error_util",
    srcs = ["utils/error_util.cc"],
    hdrs = ["utils/error_util.h"],
    copts = ["-std=c++17"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "tf_dialect_passes",
    srcs = [
        "transforms/constant_fold.cc",
        "transforms/decode_constant.cc",
        "transforms/dialect_hooks.cc",
    ],
    hdrs = [
        "transforms/constant_fold.h",
        "transforms/decode_constant.h",
    ],
    copts = ["-std=c++17"],
    deps = [
        ":convert_tensor",
        ":eval_util",
        ":tensorflow",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/core:framework",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_dialect_lib",
    copts = ["-std=c++17"],
    deps = [
        ":tensorflow_dialect_registration",
        ":tf_dialect_passes",
        "@local_config_mlir//:StandardDialectRegistration",
    ],
)

cc_library(
    name = "tf_graph_optimization_pass",
    srcs = ["transforms/tf_graph_optimization_pass.cc"],
    copts = ["-std=c++17"],
    deps = [
        ":convert_graphdef",
        ":mlir_roundtrip_flags",
        ":mlir_roundtrip_pass",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow_pass_registration",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "eval_util",
    srcs = ["utils/eval_util.cc"],
    hdrs = ["utils/eval_util.h"],
    copts = ["-std=c++17"],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":export_tf_dialect_op",
        ":mangling_util",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Support",
    ],
)

cc_library(
    name = "translate_lib",
    srcs = [
        "translate/tf_mlir_translate.cc",
    ],
    hdrs = [
        "translate/tf_mlir_translate.h",
    ],
    copts = ["-std=c++17"],
    deps = [
        ":convert_graphdef",
        ":error_util",
        ":import_utils",
        ":mangling_util",
        ":mlir_roundtrip_flags",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Pass",
    ],
)

cc_library(
    name = "translate_cl_options",
    srcs = [
        "translate/tf_mlir_translate_cl.cc",
    ],
    hdrs = [
        "translate/tf_mlir_translate_cl.h",
    ],
    copts = ["-std=c++17"],
    deps = [
        "@llvm//:support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "translate_registration",
    srcs = [
        "translate/tf_mlir_translate_registration.cc",
    ],
    copts = ["-std=c++17"],
    deps = [
        ":convert_graphdef",
        ":mlir_roundtrip_flags",
        ":translate_cl_options",
        ":translate_lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Translation",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "error_util_test",
    srcs = ["utils/error_util_test.cc"],
    tags = ["no_rocm"],
    deps = [
        ":error_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm//:support",
        "@local_config_mlir//:IR",
    ],
)

tf_native_cc_binary(
    name = "derived_attr_populator_gen",
    srcs = [
        "translate/derived_attr_populator_gen.cc",
    ],
    deps = [
        "@llvm//:support",
        "@llvm//:tablegen",
        "@local_config_mlir//:TableGen",
    ],
)

genrule(
    name = "derived_attr_populator_inc",
    srcs = [
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
        "ir/tf_generated_ops.td",
        "ir/tf_op_base.td",
        "ir/tf_ops.td",
    ],
    outs = [
        "translate/derived_attr_populator.inc",
    ],
    cmd = ("$(location :derived_attr_populator_gen) " +
           "-I external/local_config_mlir/include " +
           "-I external/org_tensorflow " +
           "$(location //tensorflow/compiler/mlir/tensorflow:ir/tf_ops.td) " + " -o $@"),
    tools = [":derived_attr_populator_gen"],
)

filegroup(
    name = "tensorflow_optimize_td_files",
    srcs = [
        "transforms/optimize.td",
    ],
)

gentbl(
    name = "tensorflow_optimize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/optimize.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        #"@local_config_mlir//:StdOpsTdFiles",
    ],
)

td_library(
    name = "DialectTdFiles",
    srcs = [
        "ir/dialect.td",
        "ir/types.td",
    ],
    includes = ["include"],
    deps = [
        "@local_config_mlir//:BuiltinDialectTdFiles",
        "@local_config_mlir//:CallInterfacesTdFiles",
        "@local_config_mlir//:ControlFlowInterfacesTdFiles",
        "@local_config_mlir//:InferTypeOpInterfaceTdFiles",
        "@local_config_mlir//:OpBaseTdFiles",
        "@local_config_mlir//:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "AttributesIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "attributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "attributes.cc.inc",
        ),
        (
            ["-gen-enum-decls"],
            "attributes_enum.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "attributes_enum.cc.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/attributes.td",
    deps = [":DialectTdFiles"],
)

gentbl_cc_library(
    name = "TypesIncGen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "tf_types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "tf_types.cc.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/types.td",
    deps = [":DialectTdFiles"],
)

gentbl_cc_library(
    name = "tf_pass_inc_gen",
    #compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorFlow",
            ],
            "transforms/tf_passes.h.inc",
        ),
#        (
#            ["-gen-pass-doc"],
#            "g3doc/_includes/tf_passes.md",
#        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/tf_passes.td",
    deps = [
        "@local_config_mlir//:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensorflow_op_interfaces_inc_gen",
    #compatible_with = get_compatible_with_portable(),
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "ir/tf_op_interfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "ir/tf_op_interfaces.cc.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/tf_op_interfaces.td",
    test = True,
    deps = [
        "@local_config_mlir//:PassBaseTdFiles",
        ":tensorflow_ops_td_files",
    ],
)


gentbl_cc_library(
    name = "DialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect",
                "tf_type",
            ],
            "ir/dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect",
                "tf_type",
            ],
            "ir/dialect.cc.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/dialect.td",
    deps = [":DialectTdFiles"],
)