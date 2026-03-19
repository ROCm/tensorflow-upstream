---
name: sync-test-check
description: Check excluded ROCm tests to see if they still fail. Runs each excluded test individually and removes passing tests from the exclusion lists in run_xla.sh and run_gpu_single.sh.
---

# Check Excluded ROCm Tests

You are verifying whether previously-excluded ROCm tests are still failing. Over time, upstream fixes or ROCm improvements may cause excluded tests to start passing. This skill runs each excluded test individually and removes any that now pass from the exclusion lists.

The exclusion lists live in two files:
- `tensorflow/tools/ci_build/linux/rocm/run_xla.sh` — XLA tests
- `tensorflow/tools/ci_build/linux/rocm/run_gpu_single.sh` — GPU/pycpp tests

This skill has helper scripts to run individual tests:
- `.claude/skills/sync-test-check/scripts/run_single_xla_test.sh <bazel_target> <test_filter>`
- `.claude/skills/sync-test-check/scripts/run_single_gpu_test.sh <bazel_target> <test_filter>`

Run all steps automatically. Each test may take several minutes to build and run.

**IMPORTANT: Tests must be run strictly one at a time (sequentially, never in parallel).** The GPU test infrastructure uses `parallel_gpu_execute`, a file-based `flock(1)` locking system that assigns each test exclusive access to a single GPU via `/var/lock/gpulock*`. The lock is **non-blocking** — if all GPU slots are occupied, the test immediately fails with "Cannot find a free GPU" rather than waiting. Running multiple `bazel test` commands simultaneously will cause false failures due to lock contention. Even the codebase notes that `parallel_gpu_execute` is "fragile". Always wait for one test to fully complete before starting the next.

## Pre-flight checks

1. Confirm the current branch matches `develop-upstream-sync-*`. If not, STOP.
2. Confirm the working tree is clean (`git status`). If not, STOP.
3. Confirm the helper scripts exist and are executable:
   ```bash
   test -x .claude/skills/sync-test-check/scripts/run_single_xla_test.sh && echo "xla runner OK"
   test -x .claude/skills/sync-test-check/scripts/run_single_gpu_test.sh && echo "gpu runner OK"
   ```

## Step 0: Find tests where `cuda-only` tag was removed in this sync

During conflict resolution, `cuda-only` tags may have been removed from Bazel test targets (because the tag was being misused to disable failing tests rather than to mark tests as truly CUDA-only). These tests are now enabled for ROCm but may fail — they must be tested and, if failing, added to the exclusion list.

### 0.1 Find removed `cuda-only` occurrences

```bash
git diff develop-upstream HEAD -- "*.BUILD" "*/BUILD" "*/BUILD.bazel" | grep -B 10 '^-.*"cuda-only"'
```

For each removed `cuda-only` line, identify the enclosing Bazel target name by reading the surrounding context (look for `name = "..."` within the same target block). Then determine the full Bazel target path from the file path (e.g., `third_party/xla/xla/stream_executor/gpu/BUILD` → `//xla/stream_executor/gpu:<target_name>` for XLA targets, or `tensorflow/...` → `//tensorflow/...:<target_name>`).

If no `cuda-only` removals are found, skip to Step 1.

### 0.2 Run each newly-enabled test

For each target identified, run it using the appropriate helper script. Use the GPU test runner for targets under `tensorflow/`, and the XLA test runner for targets under `third_party/xla/`:

```
Tool: Bash
command: .claude/skills/sync-test-check/scripts/run_single_gpu_test.sh "<bazel_target>" "" 2>&1 | tail -50
run_in_background: true
```

Wait for each test to complete before starting the next (same sequential rule as Step 2).

### 0.3 Handle results

- **PASS**: The test works on ROCm — no action needed. Report it as enabled and passing.
- **FAIL / BUILD_ERROR**: The test fails on ROCm — add it to the `EXCLUDED_TESTS` array in `tensorflow/tools/ci_build/linux/rocm/run_gpu_single.sh` with a comment identifying the target, then commit immediately:
  ```bash
  git add -u
  git commit -m "Exclude failing test newly enabled by cuda-only tag removal: <target>"
  ```

## Step 1: Parse the exclusion lists

Read both test scripts and extract every excluded test entry along with its associated bazel target.

### 1.1 Parse `run_xla.sh`

Read `tensorflow/tools/ci_build/linux/rocm/run_xla.sh` and extract the `EXCLUDED_TESTS` array. Each entry has:
- A **comment line** above it (or group of entries) with the bazel target, e.g. `# @xla//xla/backends/gpu/codegen/triton:triton_gemm_fusion_test_amdgpu_any`
- One or more **test filter names**, e.g. `CompareTest.SplitK`

Also check for **whole-file exclusions** at the end of the bazel command (lines starting with `-@xla//` or `-//`). These are entire bazel targets excluded from the test suite.

Build a list of `(bazel_target, test_filter, source_file="xla")` tuples.

### 1.2 Parse `run_gpu_single.sh`

Read `tensorflow/tools/ci_build/linux/rocm/run_gpu_single.sh` and extract the `EXCLUDED_TESTS` array using the same comment→filter pattern.

Build a list of `(bazel_target, test_filter, source_file="gpu")` tuples.

### 1.3 Report the full list

Print the complete list of excluded tests with their targets, grouped by source file. Show the total count.

## Step 2: Run, update, and commit — incrementally

**CRITICAL: Update the exclusion list and commit after EACH passing test, not at the end.** The exclusion list is long and context may be exhausted before all tests are checked. By committing after each pass, progress is never lost — if the session ends mid-way, all previously-verified results are already saved in git history.

### 2.1 Process each test — ONE AT A TIME

For each `(bazel_target, test_filter, source_file)` tuple from Step 1, repeat this loop:

#### A. Run the test

**For XLA tests** (`source_file="xla"`):
```
Tool: Bash
command: .claude/skills/sync-test-check/scripts/run_single_xla_test.sh "<bazel_target>" "<test_filter>" 2>&1 | tail -50
run_in_background: true
```

**For GPU tests** (`source_file="gpu"`):
```
Tool: Bash
command: .claude/skills/sync-test-check/scripts/run_single_gpu_test.sh "<bazel_target>" "<test_filter>" 2>&1 | tail -50
run_in_background: true
```

**For whole-file exclusions** (entire bazel targets excluded at the command line, not in `EXCLUDED_TESTS` array):
```
Tool: Bash
command: .claude/skills/sync-test-check/scripts/run_single_xla_test.sh "<bazel_target>" "" 2>&1 | tail -50
run_in_background: true
```

Wait for the test to complete before doing anything else. Use `TaskOutput` (block: true, timeout: 600000) — individual tests should not take more than 10 minutes.

#### B. Classify the result

- **PASS**: Exit code 0 — the test no longer fails
- **FAIL**: Non-zero exit code — the test still fails, keep it excluded
- **BUILD_ERROR**: Failed to build — keep it excluded
- **TIMEOUT**: Did not complete within 10 minutes — keep it excluded

#### C. If PASS: immediately remove from exclusion list and commit

1. **Edit** the source file (`run_xla.sh` or `run_gpu_single.sh`):
   - Remove the passing test filter line from the `EXCLUDED_TESTS` array
   - If this was the last filter under a comment (bazel target), remove the comment line too
   - If a whole-file exclusion now passes, remove the `-@xla//...` line from the bazel command
   - Preserve blank lines and formatting conventions of the file
   - Do NOT remove comment-only lines that are section headers (like `# vvv TODO (rocm) weekly-sync-XXXXXXXX excluded tests`)

2. **Verify** the script is still valid bash:
   ```bash
   bash -n tensorflow/tools/ci_build/linux/rocm/<modified-script>.sh
   ```

3. **Commit** immediately:
   ```bash
   git add -u
   git commit -m "Remove passing excluded test: <test_filter>"
   ```
   Use `git add -u` (not `git add -A`) to avoid staging untracked files.

This way, each passing test is persisted in a separate commit. If the session ends unexpectedly, all verified results are safe.

#### D. If FAIL/BUILD_ERROR/TIMEOUT: report and move on

Report the result to the user (test name, status, brief reason) and proceed to the next test. No edits needed.

### 2.2 Group tests by target for efficiency

When multiple excluded test filters belong to the same bazel target, you can test them together in a single run by combining filters with `:`:
```
Tool: Bash
command: .claude/skills/sync-test-check/scripts/run_single_xla_test.sh "<bazel_target>" "<filter1>:<filter2>:<filter3>" 2>&1 | tail -100
run_in_background: true
```

If the combined run passes, all tests in the group pass — remove them all and commit:
```bash
git add -u
git commit -m "Remove passing excluded tests from <target_short_name>"
```

If the combined run fails, re-run each filter individually to identify which ones still fail and which now pass. Remove and commit each passing one individually.

### 2.3 Track progress

After each test (pass or fail), report a running tally to the user:
```
Progress: X/Y tests checked. P passed (removed), F still failing.
```

## Step 3: Final report

After all tests have been checked (or if the session must end early):

**Tests checked:** X total (Y from run_xla.sh, Z from run_gpu_single.sh)

**Results:**

| Status | Count |
|--------|-------|
| PASS (removed & committed) | N |
| FAIL (kept) | N |
| BUILD_ERROR (kept) | N |
| TIMEOUT (kept) | N |

**Tests removed (now passing):**

| Test filter | Bazel target | Source | Commit |
|------------|-------------|--------|--------|
| CompareTest.SplitK | @xla//...:triton_gemm_fusion_test_amdgpu_any | run_xla.sh | abc1234 |

**Tests still failing (kept):**

| Test filter | Bazel target | Failure reason |
|------------|-------------|----------------|
| TopKTests/TopKKernelTest.* | @xla//...:topk_test_amdgpu_any | Assertion failed: expected X got Y |

**If not all tests were checked** (session ended early), report:
- How many tests remain unchecked
- "Run `/sync-test-check` again to continue — already-removed tests will not be re-tested since they are no longer in the exclusion list."

## Important notes
- Do NOT push anything. The user will push when ready.
- Do NOT modify any test source code. This skill only modifies the exclusion lists.
- Tests that use wildcards (e.g. `TopKTests/TopKKernelTest.*`) should be run as-is — the wildcard is part of the gtest filter syntax.
- If a test is marked with a comment like `# failing on mi250`, note that the result may depend on the GPU hardware being used. Report the hardware detected.
- NEVER run tests in parallel. The `parallel_gpu_execute` flock mechanism is non-blocking and will cause false "Cannot find a free GPU" failures if multiple tests compete for GPU locks.
- For very large exclusion lists, report progress periodically so the user knows work is happening.

### Parameterized test patterns

When a passing test is removed from the exclusion list, check whether the remaining exclusion entries for the same target are **complete**. A common error is having a pattern like `SortRewriterTest.*` that only covers non-parameterized `TEST_F` tests, while `TEST_P` tests with names like `SortRewriterTest/SortRewriterTest.SortNumpyOrder/bf16_desc` are not covered by any pattern.

If after removing a passing entry the target still shows as `FAIL` in the RBE run (i.e. the test binary exits non-zero despite the exclusion), the remaining patterns may not be covering all failing cases. In that situation, re-read the test log and check whether uncovered parameterized test names are slipping through — then add the missing `InstantiationName/SuiteName.*` patterns.
