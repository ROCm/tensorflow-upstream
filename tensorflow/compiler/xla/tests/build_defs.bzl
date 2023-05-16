"""Build rules for XLA testing."""

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/compiler/xla/tests:plugin.bzl", "plugins")
load(
    "//tensorflow/compiler/xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

all_backends = ["cpu", "gpu"] + plugins.keys()

def xla_test(
        name,
        srcs,
        deps,
        xla_test_library_deps = [],
        backends = [],
        disabled_backends = [],
        real_hardware_only = False,
        args = [],
        tags = [],
        copts = [],
        data = [],
        backend_tags = {},
        backend_args = {},
        **kwargs):
    """Generates cc_test targets for the given XLA backends.

    This rule generates a cc_test target for one or more XLA backends. The arguments
    are identical to cc_test with two additions: 'backends' and 'backend_args'.
    'backends' specifies the backends to generate tests for ("cpu", "gpu"), and
    'backend_args'/'backend_tags' specifies backend-specific args parameters to use
    when generating the cc_test.

    The name of the cc_tests are the provided name argument with the backend name
    appended. For example, if name parameter is "foo_test", then the cpu
    test target will be "foo_test_cpu".

    The build rule also defines a test suite ${name} which includes the tests for
    each of the supported backends.

    Each generated cc_test target has a tag indicating which backend the test is
    for. This tag is of the form "xla_${BACKEND}" (eg, "xla_cpu"). These
    tags can be used to gather tests for a particular backend into a test_suite.

    Examples:

      # Generates the targets: foo_test_cpu and foo_test_gpu.
      xla_test(
          name = "foo_test",
          srcs = ["foo_test.cc"],
          backends = ["cpu", "gpu"],
          deps = [...],
      )

      # Generates the targets: bar_test_cpu and bar_test_gpu. bar_test_cpu
      # includes the additional arg "--special_cpu_flag".
      xla_test(
          name = "bar_test",
          srcs = ["bar_test.cc"],
          backends = ["cpu", "gpu"],
          backend_args = {"cpu": ["--special_cpu_flag"]}
          deps = [...],
      )

    The build rule defines the preprocessor macro XLA_TEST_BACKEND_${BACKEND}
    to the value 1 where ${BACKEND} is the uppercase name of the backend.

    Args:
      name: Name of the target.
      srcs: Sources for the target.
      deps: Dependencies of the target.
      xla_test_library_deps: If set, the generated test targets will depend on the
        respective cc_libraries generated by the xla_test_library rule.
      backends: A list of backends to generate tests for. Supported values: "cpu",
        "gpu". If this list is empty, the test will be generated for all supported
        backends.
      disabled_backends: A list of backends to NOT generate tests for.
      args: Test arguments for the target.
      tags: Tags for the target.
      copts: Additional copts to pass to the build.
      data: Additional data to pass to the build.
      backend_tags: A dict mapping backend name to list of additional tags to
        use for that target.
      backend_args: A dict mapping backend name to list of additional args to
        use for that target.
      **kwargs: Additional keyword arguments to pass to native.cc_test.
    """

    # All of the backends in all_backends are real hardware.
    _ignore = [real_hardware_only]

    test_names = []
    if not backends:
        backends = all_backends

    backends = [
        backend
        for backend in backends
        if backend not in disabled_backends
    ]

    for backend in backends:
        test_name = "%s_%s" % (name, backend)
        this_backend_tags = ["xla_%s" % backend]
        this_backend_copts = []
        this_backend_args = backend_args.get(backend, [])
        this_backend_data = []
        if backend == "cpu":
            backend_deps = ["//tensorflow/compiler/xla/service:cpu_plugin"]
            backend_deps += ["//tensorflow/compiler/xla/tests:test_macros_cpu"]
        elif backend == "gpu":
            backend_deps = if_gpu_is_configured(["//tensorflow/compiler/xla/service:gpu_plugin"])
            backend_deps += if_gpu_is_configured(["//tensorflow/compiler/xla/tests:test_macros_gpu"])
            this_backend_tags += tf_gpu_tests_tags()
        elif backend in plugins:
            backend_deps = []
            backend_deps += plugins[backend]["deps"]
            this_backend_copts += plugins[backend]["copts"]
            this_backend_tags += plugins[backend]["tags"]
            this_backend_args += plugins[backend]["args"]
            this_backend_data += plugins[backend]["data"]
        else:
            fail("Unknown backend %s" % backend)

        if xla_test_library_deps:
            for lib_dep in xla_test_library_deps:
                backend_deps += ["%s_%s" % (lib_dep, backend)]

        tf_cc_test(
            name = test_name,
            srcs = srcs,
            tags = tags + backend_tags.get(backend, []) + this_backend_tags,
            extra_copts = copts + ["-DXLA_TEST_BACKEND_%s=1" % backend.upper()] +
                          this_backend_copts,
            args = args + this_backend_args,
            deps = deps + backend_deps,
            data = data + this_backend_data,
            **kwargs
        )

        test_names.append(test_name)

    native.test_suite(name = name, tags = tags, tests = test_names)

def xla_test_library(
        name,
        srcs,
        hdrs = [],
        deps = [],
        backends = []):
    """Generates cc_library targets for the given XLA backends.

    This rule forces the sources to be compiled for each backend so that the
    backend specific macros could expand correctly. It's useful when test targets
    in different directories referring to the same sources but test with different
    arguments.

    Examples:

      # Generates the targets: foo_test_library_cpu and foo_test_gpu.
      xla_test_library(
          name = "foo_test_library",
          srcs = ["foo_test.cc"],
          backends = ["cpu", "gpu"],
          deps = [...],
      )
      # Then use the xla_test rule to generate test targets:
      xla_test(
          name = "foo_test",
          srcs = [],
          backends = ["cpu", "gpu"],
          deps = [...],
          xla_test_library_deps = [":foo_test_library"],
      )

    Args:
      name: Name of the target.
      srcs: Sources for the target.
      hdrs: Headers for the target.
      deps: Dependencies of the target.
      backends: A list of backends to generate libraries for.
        Supported values: "cpu", "gpu". If this list is empty, the
        library will be generated for all supported backends.
    """

    if not backends:
        backends = all_backends

    for backend in backends:
        this_backend_copts = []
        if backend in ["cpu", "gpu"]:
            backend_deps = ["//tensorflow/compiler/xla/tests:test_macros_%s" % backend]
        elif backend in plugins:
            backend_deps = plugins[backend]["deps"]
            this_backend_copts += plugins[backend]["copts"]
        else:
            fail("Unknown backend %s" % backend)

        native.cc_library(
            name = "%s_%s" % (name, backend),
            srcs = srcs,
            testonly = True,
            hdrs = hdrs,
            copts = ["-DXLA_TEST_BACKEND_%s=1" % backend.upper()] +
                    this_backend_copts,
            deps = deps + backend_deps,
        )

def generate_backend_suites(backends = []):
    if not backends:
        backends = all_backends
    for backend in backends:
        native.test_suite(
            name = "%s_tests" % backend,
            tags = ["xla_%s" % backend, "-broken", "manual"],
        )

def generate_backend_test_macros(backends = []):
    if not backends:
        backends = all_backends
    for backend in backends:
        manifest = ""
        if backend in plugins:
            manifest = plugins[backend]["disabled_manifest"]

        native.cc_library(
            name = "test_macros_%s" % backend,
            testonly = True,
            srcs = ["test_macros.cc"],
            hdrs = ["test_macros.h"],
            copts = [
                "-DXLA_PLATFORM=\\\"%s\\\"" % backend.upper(),
                "-DXLA_DISABLED_MANIFEST=\\\"%s\\\"" % manifest,
            ],
            deps = [
                "//tensorflow/tsl/platform:logging",
            ],
        )
