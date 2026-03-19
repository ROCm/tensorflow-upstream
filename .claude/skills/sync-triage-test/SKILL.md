---
name: sync-triage-test
description: Investigates a failing XLA RBE test on MI250 (gfx90a), determines the root cause, and either fixes the code or adds the test to the exclusion list in run_xla.sh. Use when user says "investigate failing test", "triage test", "fix failing test", "why is <test> failing", or provides a specific failing test target from an RBE run.
---

# Triage a Failing XLA RBE Test

You are investigating a specific failing test from the MI250 RBE test run. Your goal is to determine the root cause and take one of two actions:

1. **Fix the code** — if the failure is due to a ROCm-specific bug, a missing HIP implementation, or a broken upstream change that can clearly and safely be corrected.
2. **Add to exclusion list** — in all other cases, including when a fix is possible but non-trivial, ambiguous, or risky.

**When in doubt, exclude. Never ask the user to decide.**

The user will provide a failing test target and optionally a test filter (test case name). If not provided, ask for both before proceeding.

Run all steps automatically without asking the user for guidance.

## Pre-flight checks

1. Confirm the helper script exists and is executable:
   ```bash
   test -x .claude/skills/sync-triage-test/scripts/run_single_xla_rbe_test.sh && echo "OK"
   ```
   If not executable, run `chmod +x .claude/skills/sync-triage-test/scripts/run_single_xla_rbe_test.sh`.

2. Confirm TLS certificates exist:
   ```bash
   test -f /tf/certificates/ci-cert.crt && test -f /tf/certificates/ci-cert.key && echo "certs OK"
   ```
   If missing, STOP and tell the user.

## Step 0: Check for partially-excluded targets

Before reproducing, check if the test target already has some (but not all) cases in the exclusion list. This is critical because the RBE run does not apply `--test_filter` — a test binary with 3 excluded cases and 1 new failing case will still show as `FAIL:` in the RBE output.

1. Read `tensorflow/tools/ci_build/linux/rocm/run_xla.sh` and find all `EXCLUDED_TESTS` entries.
2. If the test log from the RBE run is available (path printed after `FAIL:`), read it and extract all `[  FAILED  ]` lines to get the specific failing case names.
3. Cross-reference: for each failing case, check if it matches an entry in `EXCLUDED_TESTS`.
4. If **all** failing cases are already excluded → report "All failing cases already excluded — no action needed" and stop.
5. If some cases are new → proceed with only the **new** (non-excluded) cases. Set the test filter to target only these cases.

This prevents the mistake of skipping a test target as "already excluded" when it actually contains new failures.

### Parameterized test name matching (critical)

gtest has two test types with **different full-name formats**:

- **`TEST_F` (non-parameterized):** `SuiteName.TestName`
  - Example: `SortRewriterTest.SortKeysLessThan`
  - Matched by pattern: `SortRewriterTest.*`

- **`TEST_P` (parameterized):** `InstantiationName/SuiteName.TestName/ParamName`
  - Example: `SortRewriterTest/SortRewriterTest.SortNumpyOrder/bf16_desc`
  - Example: `NumericTestsForBlas/NumericTestsForBlas.Infinity/dot_bf16_bf16_f32_x9`
  - Example: `SortRewriterArgsort/SortRewriterArgsortTest.SortNumpyOrderArgsort/f16_desc_s32_cub`

**The pattern `SuiteName.*` does NOT match `InstantiationName/SuiteName.TestName/ParamName`** because `SuiteName.*` requires a dot immediately after `SuiteName`, but parameterized names have a slash.

When cross-referencing failures against exclusion patterns, check: does the existing pattern actually match the failing test's full name? A target with `SortRewriterTest.*` in exclusions is **not** covered for `SortRewriterTest/SortRewriterTest.SortNumpyOrder/bf16_desc`.

## Step 1: Reproduce the failure

**For build failures (FAILED TO BUILD):** Do NOT re-run the test — it will just fail to build again. Instead, extract the compiler error from the RBE run output (search for `error:` lines near the target name). Then go directly to Step 2 → classify as Category E → Step 3 → investigate the source and fix the build error. After applying the fix, re-run via RBE to verify the build and test pass.

**For test failures (FAIL):** Run the failing test via RBE to get a fresh, detailed failure log:

```
Tool: Bash
command: .claude/skills/sync-triage-test/scripts/run_single_xla_rbe_test.sh "<bazel_target>" "<test_filter>" 2>&1
run_in_background: true
```

Wait for completion with `TaskOutput` (block: true, timeout: 600000).

If the test **passes** on re-run, report: "Test passed on re-run — likely an intermittent failure (pool contention, flaky test, or transient hardware error). No action taken." Then stop.

## Step 2: Classify the failure

Read the full test output and classify into one of these categories:

### A. Infrastructure / environment error
Symptoms: linker errors, missing `.so`, `GLIBC_*` not found, `HIP_ERROR_OutOfMemory` at startup, `cannot open shared object`.
→ This is not a test failure. Report the infrastructure issue to the user. Do NOT add to exclusion list. Stop.

### B. Unsupported op / CUDA-only feature
Symptoms: `Unimplemented`, `not supported on ROCm`, `CUDA-only`, hipBLASLt/cuDNN-specific error, `CUBLAS_STATUS_*`, or **platform name mismatch** (`Could not find registered platform with name: "CUDA". Available platform names are: ROCM`).
→ Add to exclusion list.

**Platform mismatch heuristic:** When the error mentions "CUDA" vs "ROCM" platform names, do NOT attempt to fix the test by replacing hardcoded `"CUDA"` with `se::GpuPlatformName()`. Instead, read the **implementation source** (not the test) and check whether the underlying feature supports ROCm. Look for patterns like:
- `if (platform_name == "CUDA")` with no ROCm branch
- `return.*Unimplemented.*platform` for non-CUDA platforms
- FFI handlers registered only for `"CUDA"` (e.g. `ffi::FindHandler(..., "CUDA")`)
- `#if GOOGLE_CUDA` blocks with no `#elif TENSORFLOW_USE_ROCM` counterpart

If the implementation is CUDA-only, the test failure is a consequence of the missing ROCm support — fixing the test to be "platform-aware" would just shift the error. Classify as Category B and exclude.

### C. Numerical / correctness failure
Symptoms: `ASSERT_NEAR failed`, `Expected: X, got: Y`, wrong output values.
→ Investigate source (Step 3). If a clear, minimal fix exists, fix it. Otherwise, exclude.

### D. Crash / HIP runtime error
Symptoms: `HIP error`, `hipErrorInvalidValue`, segfault, `Aborted`.
→ Investigate source (Step 3). If a clear, minimal fix exists, fix it. Otherwise, exclude.

### E. Build / compilation failure
Symptoms: `FAILED TO BUILD`, `ERROR: Build did NOT complete`, compiler error (`error: use of undeclared identifier`, `error: no member named`), missing symbol, missing header.
→ Fix the build error directly (not an exclusion). Build failures are often caused by:
- **Missed renames during merge**: upstream renamed a class/function (e.g. `AutotunerUtil` → `AutotunerCache`) but the merge re-introduced old references from a conflicting upstream commit. Check `git log` on the file and the upstream commits to find the rename.
- **Missing `#include`**: a new upstream commit uses a symbol without including the right header.
- **ROCm-specific `#ifdef` conflicts**: merge brought in CUDA code inside a ROCm guard or vice versa.

For build failures, always use `git log` and `git show` to reference the specific commits that caused the conflict in your commit message.

### F. Test logic failure / assertion
Symptoms: `FAIL` with a clear assertion in the test body.
→ Investigate source (Step 3). If a clear, minimal fix exists, fix it. Otherwise, exclude.

## Step 3: Investigate the source

For categories C, D, F — read the relevant source files before deciding:

1. Find the test source from the bazel target (e.g. `@xla//xla/backends/gpu/runtime:foo_test` → `third_party/xla/xla/backends/gpu/runtime/foo_test.cc`).
2. **Always read the implementation under test before deciding on a fix.** A test may look like it just needs a string change, but if the underlying implementation doesn't support ROCm, the test failure is a symptom — not the disease. For example, if a test hardcodes `"CUDA"` but the implementation's `::Create()` method returns `UnimplementedError` for non-CUDA platforms, making the test platform-aware would just produce a different error.
3. Look for `#if TENSORFLOW_USE_ROCM` / `#if GOOGLE_CUDA` guards — a missing ROCm path explains many failures.
4. Check recent upstream changes:
   ```bash
   git log --oneline -20 -- <file_path>
   ```
5. If a CUDA implementation exists without a ROCm equivalent, look for similar ROCm implementations nearby.

**Fix only if:** the fix is minimal, safe, and clearly correct. If there is any ambiguity — exclude instead.

## Step 4: Act and commit

Perform exactly **one commit per test**, regardless of whether the action is a fix or an exclusion. The commit message must contain the full analysis.

### If fixing the code

Apply the minimal fix:
- Preserve `#if TENSORFLOW_USE_ROCM` blocks and ROCm-specific logic.
- Mirror CUDA patterns in the ROCm path.
- No refactoring, no cleanup, no new comments outside the fix.

Re-run the test to verify the fix:
```
Tool: Bash
command: .claude/skills/sync-triage-test/scripts/run_single_xla_rbe_test.sh "<bazel_target>" "<test_filter>" 2>&1
run_in_background: true
```

If it still fails after the fix, discard the fix (`git checkout -- .`) and fall back to excluding instead.

Commit with the full analysis in the message:
```bash
git add -u
git commit -m "$(cat <<'EOF'
Fix <TestClass.TestCase> for ROCm

Test: <bazel_target>
Filter: <test_filter>
Category: <category description, e.g. "Build / compilation failure", "Unsupported op / CUDA-only feature">

Root cause:
<One or two sentences describing why the test was failing.>

Fix:
<What was changed and why — file(s), what the old code did, what the new code does.>

Verified: test passes on MI250 (gfx90a) via RBE after fix.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

### If adding to the exclusion list

Add the failing test filter to the `EXCLUDED_TESTS` array in `tensorflow/tools/ci_build/linux/rocm/run_xla.sh`.

**Every `[  FAILED  ]` line in the test output must be covered by an exclusion pattern.** Before committing, re-read all `[  FAILED  ]` lines from the test log and verify each one matches at least one pattern in `EXCLUDED_TESTS`. It is not sufficient to add a pattern that covers the first failure — if the test binary reports 20 failures, all 20 must be matched. Use wildcards (`*`) to cover groups, but confirm the wildcard actually matches the full test name format (see below).

### Choosing the right exclusion pattern

The gtest filter pattern must match the **full test name** as it appears in `[  FAILED  ]` lines. Always copy the exact name from the test output and derive the pattern from it.

**Non-parameterized tests (`TEST_F`)** — full name is `SuiteName.TestName`:
```bash
    FailingTestSuite.FailingTestCase
    FailingTestSuite.*          # excludes all TEST_F cases in the suite
```

**Parameterized tests (`TEST_P`)** — full name is `InstantiationName/SuiteName.TestName/ParamName`:
```bash
    # To exclude a specific parameter:
    NumericTestsForBlas/NumericTestsForBlas.Infinity/dot_bf16_bf16_f32_x9

    # To exclude all test methods for a specific parameter:
    NumericTestsForBlas/NumericTestsForBlas.*/dot_bf16_bf16_f32_x9

    # To exclude all parameters for a specific instantiation:
    SortRewriterTest/SortRewriterTest.*
    SortRewriterArgsort/SortRewriterArgsortTest.*
```

**A suite may have both `TEST_F` and `TEST_P` tests.** If both kinds are failing, add patterns for both:
```bash
    SortRewriterTest.*                     # covers TEST_F cases
    SortRewriterTest/SortRewriterTest.*    # covers TEST_P cases (parameterized)
```

**Never use `SuiteName.*` alone to exclude a `TEST_P` test** — it will not match.

Format (match existing style exactly):
```bash
    # @xla//xla/backends/...:some_test_amdgpu_any
    FailingTestClass.FailingTestCase
```

Placement: append under the most recent `# vvv TODO (rocm) weekly-sync-XXXXXX excluded tests` section header. If there is no header for the current sync, add one first (derive the date from the branch name `develop-upstream-sync-XXXXXX` or from `git log --oneline -1`):
```bash
    # vvv TODO (rocm) weekly-sync-XXXXXX excluded tests

    # @xla//xla/...:some_test_amdgpu_any
    FailingTestClass.FailingTestCase
```

Verify the script is still valid bash:
```bash
bash -n tensorflow/tools/ci_build/linux/rocm/run_xla.sh
```

**If the original failure was a crash (segfault, `Aborted`, `HIP error`, `hipErrorInvalidValue`, or any signal-based termination), you MUST re-run the full test binary via RBE after adding the exclusion patterns.** A crash aborts the entire test process mid-run, hiding all test cases that would have run after the crash. Once those cases are excluded, a new crash in a previously-hidden case may surface. This is a cascading pattern: excluding x9 reveals x6 crashes, excluding x6 reveals x3 crashes, and so on.

Run the full test binary without a test filter to let the updated global exclusion list take effect:

```
Tool: Bash
command: .claude/skills/sync-triage-test/scripts/run_single_xla_rbe_test.sh "<bazel_target>" "" 2>&1
run_in_background: true
```

Wait for completion with `TaskOutput` (block: true, timeout: 600000).

- If the re-run **passes**: proceed to commit.
- If the re-run **reveals a new crash**: add those new cases to `EXCLUDED_TESTS` and re-run again. Repeat until the binary passes.
- If the re-run **fails with the same test**: the exclusion pattern is wrong — check the exact `[  FAILED  ]` name against the pattern (see parameterized test naming rules above).

For non-crash failures (assertion failures, wrong output values, `FAIL` without a signal), this re-run is not required — those failures do not abort the process and all failing test names are visible in the original log.

Commit with the full analysis in the message:
```bash
git add -u
git commit -m "$(cat <<'EOF'
Exclude failing test: <TestClass.TestCase>

Test: <bazel_target>
Filter: <test_filter>
Category: <category description, e.g. "Unsupported op / CUDA-only feature", "Numerical / correctness failure">

Root cause:
<One or two sentences describing why the test is failing.>

Reason for exclusion (not fix):
<Why a fix was not applied — e.g. "CUDA-only feature (hipBLASLt not supported)",
"Numerical precision divergence in fp16 reduction, non-trivial to fix",
"Missing ROCm equivalent for cuDNN fused op", etc.>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

## Step 5: Report to user

After committing, report a concise summary:

**Test:** `<bazel_target>` / `<test_filter>`
**Category:** `<category description>`
**Root cause:** one sentence
**Action:** `FIXED` | `EXCLUDED` | `INFRASTRUCTURE` | `FLAKY`
**Commit:** `<hash> <subject>`

## Important notes

- Run tests one at a time. Never launch parallel RBE test runs.
- Do NOT push anything. The user will push when ready.
- Never ask the user for guidance on fix vs. exclude — default to exclude when uncertain.
- Each test gets exactly one commit containing the full analysis in the commit message.
- If multiple test cases from the same target are failing, run them together with `CaseA:CaseB` for efficiency. If the combined run fails, re-run individually to isolate. Each case that requires a separate action gets its own commit.
- **Never add `cuda-only` tags to BUILD files.** The correct action for CUDA-only tests is always to add the failing test cases to the exclusion list in `run_xla.sh`. The `cuda-only` tag validation is handled separately by the `sync-cuda-only` skill.
