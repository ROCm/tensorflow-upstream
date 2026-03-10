# Tools and utilities that aid in XLA development and usage.

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load(
    "@local_tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_xla//xla:xla.bzl",
    "xla_cc_binary",
    "xla_cc_test",
    "xla_internal",
    "xla_py_proto_library",
)
load("@local_xla//xla/stream_executor:build_defs.bzl", "if_gpu_is_configured")
load("@local_xla//xla/tests:build_defs.bzl", "xla_test")
load(
    "@local_xla//xla/tsl:tsl.bzl",
    "if_cuda_or_rocm",
    "if_google",
    "tsl_gpu_library",
)
load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["@local_xla//xla:internal"],
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["@local_xla//xla:internal"],
)

build_test(
    name = "hex_floats_to_packed_literal_build_test",
    targets = [
        ":hex_floats_to_packed_literal",
    ],
)

xla_cc_binary(
    name = "hex_floats_to_packed_literal",
    srcs = ["hex_floats_to_packed_literal.cc"],
    deps = [
        "@local_xla//xla/tsl/lib/io:buffered_inputstream",
        "@local_xla//xla/tsl/lib/io:random_inputstream",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

build_test(
    name = "show_signature_build_test",
    targets = [
        ":show_signature",
    ],
)

xla_cc_binary(
    name = "show_signature",
    srcs = ["show_signature.cc"],
    deps = [
        "@local_xla//xla:shape_util",
        "@local_xla//xla:types",
        "@local_xla//xla/client",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/service:interpreter_plugin",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

build_test(
    name = "show_literal_build_test",
    targets = [
        ":show_literal",
    ],
)

xla_cc_binary(
    name = "show_literal",
    srcs = ["show_literal.cc"],
    deps = [
        "@local_xla//xla:literal",
        "@local_xla//xla:types",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

build_test(
    name = "convert_computation_build_test",
    targets = [
        ":convert_computation",
    ],
)

xla_cc_binary(
    name = "convert_computation",
    srcs = ["convert_computation.cc"],
    deps = [
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
    ],
)

build_test(
    name = "hlo_module_metadata_processor_build_test",
    targets = [
        ":hlo_module_metadata_processor",
    ],
)

xla_cc_binary(
    name = "hlo_module_metadata_processor",
    srcs = ["hlo_module_metadata_processor.cc"],
    deps = [
        "@local_xla//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
    ],
)

build_test(
    name = "show_text_literal_build_test",
    targets = [
        ":show_text_literal",
    ],
)

xla_cc_binary(
    name = "show_text_literal",
    srcs = ["show_text_literal.cc"],
    deps = [
        "@local_xla//xla:literal",
        "@local_xla//xla:text_literal_reader",
        "@local_xla//xla:types",
        "@local_xla//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

build_test(
    name = "dumped_computation_to_text_build_test",
    targets = [
        ":dumped_computation_to_text",
    ],
)

xla_cc_binary(
    name = "dumped_computation_to_text",
    srcs = ["dumped_computation_to_text.cc"],
    deps = [
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:executable_build_options",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/service:interpreter_plugin",
        "@local_xla//xla/service:local_service",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

build_test(
    name = "dumped_computation_to_operation_list_build_test",
    targets = [
        ":dumped_computation_to_operation_list",
    ],
)

xla_cc_binary(
    name = "dumped_computation_to_operation_list",
    srcs = ["dumped_computation_to_operation_list.cc"],
    deps = [
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:executable_build_options",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/service:interpreter_plugin",
        "@local_xla//xla/service:local_service",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

build_test(
    name = "hlo_proto_to_json_build_test",
    targets = [
        ":hlo_proto_to_json",
    ],
)

xla_cc_binary(
    name = "hlo_proto_to_json",
    srcs = ["hlo_proto_to_json.cc"],
    deps = [
        "@local_xla//xla:util",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "hlo_extractor_test",
    srcs = ["hlo_extractor_test.cc"],
    deps = [
        ":hlo_extractor",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/utils:hlo_matchers",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_xla//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_extractor",
    srcs = ["hlo_extractor.cc"],
    hdrs = ["hlo_extractor.h"],
    deps = [
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:compilation_environments",
        "@local_xla//xla/service:hlo_module_config",
        "@local_xla//xla/service:hlo_verifier",
        "@local_xla//xla/tests:test_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "hlo-expand",
    testonly = True,
    deps = [
        ":hlo_expand_main",
    ],
)

xla_cc_binary(
    name = "hlo-opt",
    testonly = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lit_lib"],
    deps = [
        "@local_xla//xla/tools/hlo_opt:opt_main",
    ],
)

cc_library(
    name = "hlo_expand_main",
    srcs = ["hlo_expand_main.cc"],
    deps = [
        ":hlo_expand_lib",
        ":hlo_module_loader",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/pass:hlo_pass_pipeline",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "hlo_expand_lib",
    srcs = ["hlo_expand.cc"],
    hdrs = ["hlo_expand.h"],
    deps = [
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/pass:hlo_pass_pipeline",
        "@local_xla//xla/service:batchnorm_expander",
        "@local_xla//xla/service:cholesky_expander",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/service:hlo_verifier",
        "@local_xla//xla/service:rng_bit_generator_expander",
        "@local_xla//xla/service:rng_expander",
        "@local_xla//xla/service:sharding_propagation",
        "@local_xla//xla/service:triangular_solve_expander",
        "@local_xla//xla/service/spmd:stateful_rng_spmd_partitioner",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)

xla_cc_test(
    name = "hlo_expand_test",
    srcs = ["tests/hlo_expand_test.cc"],
    data = [
        "tests/cholesky.hlo",
        "tests/invalid_concat.hlo",
        "tests/spmd.hlo",
        ":hlo-expand",
    ],
    tags = [
        "nomsan",  # No msan for precompiled Nvidia binaries.
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:subprocess",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "hlo_slicer_test",
    srcs = ["hlo_slicer_test.cc"],
    deps = [
        ":hlo_slicer",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/utils:hlo_matchers",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_xla//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_slicer",
    srcs = ["hlo_slicer.cc"],
    hdrs = ["hlo_slicer.h"],
    deps = [
        ":hlo_extractor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:call_graph",
        "@local_xla//xla/service:hlo_verifier",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "interactive_graphviz",
    srcs = ["interactive_graphviz.cc"],
    deps = [
        ":hlo_extractor",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/service:compiler",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/service:hlo_runner",
        "@local_xla//xla/service:local_service",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:subprocess",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ] + if_cuda_or_rocm([
        "@local_xla//xla/service:gpu_plugin",
    ]) + if_cuda([
        "@local_xla//xla/stream_executor:cuda_platform",
    ]) + xla_internal(
        ["service:hlo_graph_dumper_google"],
        otherwise = ["@local_xla//xla/service:hlo_graph_dumper"],
    ),
)

xla_cc_test(
    name = "interactive_graphviz_bin_test",
    srcs = ["interactive_graphviz_bin_test.cc"],
    data = [
        "data/add.hlo",
        ":interactive_graphviz",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:subprocess",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "hlo_module_loader",
    srcs = ["hlo_module_loader.cc"],
    hdrs = ["hlo_module_loader.h"],
    visibility = ["@local_xla//xla:friends"],
    deps = [
        ":run_hlo_module_proto_cc",
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_module_config",
        "@local_xla//xla/service:hlo_parser",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "hlo_module_loader_test",
    srcs = ["hlo_module_loader_test.cc"],
    deps = [
        ":hlo_module_loader",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_xla//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "prepare_reference_module",
    srcs = ["prepare_reference_module.cc"],
    hdrs = ["prepare_reference_module.h"],
    deps = [
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:despecializer",
        "@local_xla//xla/service:hlo_module_config",
        "@local_xla//xla/service:hlo_runner_interface",
        "@local_xla//xla/stream_executor:platform",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "prepare_reference_module_test",
    srcs = ["prepare_reference_module_test.cc"],
    deps = [
        ":prepare_reference_module",
        "@local_xla//xla:test",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

tf_proto_library(
    name = "run_hlo_module_proto",
    srcs = ["run_hlo_module.proto"],
    protodeps = [
        "@local_xla//xla:xla_data_proto",
    ],
    visibility = ["//visibility:public"],
)

xla_py_proto_library(
    name = "run_hlo_module_pb2",
    api_version = 2,
    visibility = ["//visibility:public"],
    deps = [":run_hlo_module_proto"],
)

cc_library(
    name = "hlo_decomposer_lib",
    srcs = ["hlo_decomposer.cc"],
    hdrs = ["hlo_decomposer.h"],
    deps = [
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:call_graph",
        "@local_xla//xla/service:compilation_environments",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "run_hlo_module_lib",
    srcs = ["run_hlo_module.cc"],
    hdrs = ["run_hlo_module.h"],
    deps = [
        ":hlo_control_flow_flattening",
        ":hlo_decomposer_lib",
        ":hlo_module_loader",
        ":prepare_reference_module",
        ":run_hlo_module_proto_cc",
        "@local_xla//xla:error_spec",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_comparison",
        "@local_xla//xla:util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_module_config",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/service:hlo_runner",
        "@local_xla//xla/service:hlo_verifier",
        "@local_xla//xla/tests:test_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "run_hlo_module_test",
    srcs = ["run_hlo_module_test.cc"],
    deps = [
        ":run_hlo_module_lib",
        ":run_hlo_module_proto_cc",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_binary(
    name = "run_hlo_module",
    testonly = True,
    srcs = ["run_hlo_module_main.cc"],
    tags = [
        "noasan",  # Exceeds linker limit.
    ],
    deps = [
        ":run_hlo_module_lib",
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:translate",
        "@local_xla//xla/hlo/translate/stablehlo_to_hlo:translate",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/service:hlo_module_config",
        "@local_xla//xla/service:hlo_runner",
        "@local_xla//xla/service:interpreter_plugin",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
    ] + if_cuda_or_rocm([
        "@local_xla//xla/service:gpu_plugin",
    ]) + if_cuda([
        "@local_xla//xla/stream_executor:cuda_platform",
    ]),
)

xla_cc_test(
    name = "run_hlo_module_bin_test",
    srcs = ["run_hlo_module_bin_test.cc"],
    data = [
        "data/add.hlo",
        "data/add_mhlo.mlir",
        "data/add_stablehlo.mlir",
        "data/input_literal_f32_2_2.pbtxt",
        "data/must_alias.hlo",
        "data/must_alias_with_sharding.hlo",
        ":run_hlo_module",
    ],
    deps = [
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_parser",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:subprocess",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "hlo_control_flow_flattening",
    srcs = ["hlo_control_flow_flattening.cc"],
    hdrs = ["hlo_control_flow_flattening.h"],
    deps = [
        "@local_xla//xla:comparison_util",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/pass:hlo_pass",
        "@local_xla//xla/service:call_graph",
        "@local_xla//xla/service:collective_ops_utils",
        "@local_xla//xla/service:hlo_dce",
        "@local_xla//xla/service:tuple_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "hlo_control_flow_flattening_test",
    srcs = ["hlo_control_flow_flattening_test.cc"],
    deps = [
        ":hlo_control_flow_flattening",
        "@local_xla//xla/hlo/utils:hlo_matchers",
        "@local_xla//xla/service:collective_ops_utils",
        "@local_xla//xla/service:despecializer",
        "@local_xla//xla/service:hlo_verifier",
        "@local_xla//xla/service/spmd:spmd_partitioner",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_xla//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
    ],
)

# This target is used to reproduce miscompiles in OSS outside of TF, and it can
# not have any dependencies apart from the standard library.
cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    tags = if_google(["nofixdeps"]),
    deps = [],
)

build_test(
    name = "compute_cost_build_test",
    targets = [
        ":compute_cost",
    ],
)

xla_cc_binary(
    name = "compute_cost",
    srcs = ["compute_cost.cc"],
    deps = [
        ":hlo_module_loader",
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/service:hlo_cost_analysis",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_binary(
    name = "extract_collective_operations",
    srcs = ["extract_collective_operations.cc"],
    deps = [
        ":hlo_decomposer_lib",
        ":hlo_module_loader",
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tsl_gpu_library(
    name = "xla_compile_lib",
    srcs = ["xla_compile_lib.cc"],
    hdrs = ["xla_compile_lib.h"],
    defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]) + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    visibility = ["//visibility:public"],
    deps = [
        ":hlo_module_loader",
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:util",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/ir:hlo_module_group",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/pjrt:mlir_to_hlo",
        "@local_xla//xla/service:compiler",
        "@local_xla//xla/service:executable",
        "@local_xla//xla/service:export_hlo",
        "@local_xla//xla/service:hlo_module_config",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/service:symbol_repository",
        "@local_xla//xla/service:xla_compile_result_proto_cc_impl",
        "@local_xla//xla/service/cpu:cpu_compiler",
        "@local_xla//xla/service/cpu:cpu_executable",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor:device_memory_allocator",
        "@local_xla//xla/stream_executor:stream_executor_memory_allocator",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:env_time",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:status_to_from_proto",
        "@local_tsl//tsl/platform:statusor",
        "@stablehlo//:register",
    ] + if_cuda_is_configured([
        "@local_xla//xla/service/gpu:nvptx_compiler",
        "@local_xla//xla/service/gpu:nvptx_compiler_impl",
    ]) + if_rocm_is_configured([
        "@local_xla//xla/service/gpu:amdgpu_compiler",
        "@local_xla//xla/service/gpu:amdgpu_compiler_impl",
    ]) + if_gpu_is_configured([
        "@local_xla//xla/service/gpu:executable_proto_cc",
        "@local_xla//xla/service/gpu:gpu_compiler",
        "@local_xla//xla/service/gpu/autotuning:autotuner_util",
        "@local_xla//xla/stream_executor/gpu:gpu_init",
        "@local_xla//xla/service/gpu:gpu_symbol_repository",
    ]) + if_google(["@com_google_protobuf//:duration_cc_proto"]),
)

xla_test(
    name = "xla_cpu_compile_lib_test",
    srcs = ["xla_cpu_compile_lib_test.cc"],
    backends = [
        "cpu",
    ],
    data = [
        ":data/add.hlo",
    ],
    deps = [
        ":xla_compile_lib",
        "@local_xla//xla:util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/service:symbol_repository",
        "@local_xla//xla/service:xla_compile_result_proto_cc_impl",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_xla//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:env_time",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
        "@local_tsl//tsl/protobuf:status_proto_cc",
    ] + if_google(["@com_google_protobuf//:duration_cc_proto"]),
)

xla_test(
    name = "xla_gpu_compile_lib_test",
    srcs = ["xla_gpu_compile_lib_test.cc"],
    backends = [
        "gpu",
    ],
    data = [
        ":data/add.hlo",
        "@local_xla//xla/service:xla_aot_compile_test_gpu_target_config.prototxt",
        "@local_xla//xla/service/gpu:gpu_compiler_test_autotune_db.textproto",
    ],
    tags = [
        "config-cuda-only",
        "cuda-only",
    ],
    deps = [
        ":xla_compile_lib",
        "@local_xla//xla:util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/service:symbol_repository",
        "@local_xla//xla/service:xla_compile_result_proto_cc_impl",
        "@local_xla//xla/service/gpu:gpu_symbol_repository",
        "@local_xla//xla/service/gpu/autotuning:autotuner_util",
        "@local_xla//xla/stream_executor:device_description_proto_cc",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_xla//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
        "@local_tsl//tsl/protobuf:status_proto_cc",
    ],
)

xla_test(
    name = "hlo_decomposer_test",
    srcs = ["hlo_decomposer_test.cc"],
    deps = [
        ":hlo_decomposer_lib",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/tests:filecheck",
        "@local_xla//xla/tests:hlo_test_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)
