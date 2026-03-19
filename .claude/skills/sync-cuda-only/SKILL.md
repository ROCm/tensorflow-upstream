---
name: sync-cuda-only
description: Validates newly added cuda-only tags from an upstream sync. Runs each affected test on ROCm and removes the tag if the test passes, preventing tests from being incorrectly hidden from ROCm CI. Use when user says "check cuda-only tags", "validate cuda-only", or as part of the sync pipeline after conflict resolution.
---

# Validate Newly Added cuda-only Tags

You are checking whether `cuda-only` tags added by upstream in this sync are genuinely justified. Upstream sometimes adds `cuda-only` to tests that actually work on ROCm — this would silently exclude them from ROCm CI. Your job is to catch and reverse that.

This is the mirror of Step 0 in `sync-test-check`, which handles *removed* `cuda-only` tags. This skill handles *added* ones.

Run all steps automatically without asking the user.

## Pre-flight checks

1. Confirm the current branch matches `develop-upstream-sync-*`. If not, STOP.
2. Confirm the helper script exists and is executable:
   ```bash
   test -x .claude/skills/sync-test-check/scripts/run_single_xla_test.sh && echo "OK"
   ```

## Step 1: Find newly added cuda-only tags

```bash
git diff develop-upstream HEAD -- "*.BUILD" "*/BUILD" "*/BUILD.bazel" | grep -A 10 '^+.*"cuda-only"'
```

For each added `cuda-only` line, identify the enclosing Bazel target by reading surrounding context (look for `name = "..."` within the same target block). Determine the full Bazel target path from the file path (e.g. `third_party/xla/xla/service/gpu/tests/BUILD` → `@xla//xla/service/gpu/tests:<target_name>`).

If no newly added `cuda-only` tags are found, report "No newly added cuda-only tags found — nothing to do." and stop.

Otherwise report the full list before proceeding.

## Step 2: Check the exclusion list first

Before inspecting source, check whether the test cases are already in the `run_xla.sh` exclusion list:

```bash
grep -A 5 "<test_name>" tensorflow/tools/ci_build/linux/rocm/run_xla.sh
```

**If ALL test cases of a target are already in the exclusion list:** the `cuda-only` tag is almost certainly being misused as a secondary disable mechanism — not as a genuine CUDA-only marker. Mark it as "must run" and proceed to Step 3 regardless of what the source says.

## Step 3: Inspect the test source

Read the test source to classify it. Use these strict criteria:

**Clearly justified — skip running only if ALL of the following are true:**
- The test directly calls CUDA C APIs (`cudnn*`, `cublas*`, `cufft*`, `cuda*`) in C++ code — NOT just references these names in string literals or FileCheck patterns
- OR the entire test logic is wrapped in `#if GOOGLE_CUDA`
- AND there is no `GTEST_SKIP()` / `is_built_with_rocm_` self-skip already present

**Must run — if ANY of the following are true:**
- The test uses `GTEST_SKIP()` with a ROCm/CUDA check (it already self-skips — `cuda-only` is redundant)
- The test uses `cuda_compute_capability()` only to select output patterns or expected values (the test body still runs on ROCm)
- The test submits HLO/computation to XLA and checks the output via `MatchOptimizedHloWithShapes`, `RunAndCompare`, or similar — even if the CHECK patterns contain cuDNN target names as string literals (the strings are patterns, not API calls; the test may fail gracefully with autotuner errors rather than crashing)
- The test contains no direct CUDA C API calls (cuDNN/cuBLAS function calls in C++ code)
- The test is already in the exclusion list (Step 2)

**Critical distinction:** `MatchOptimizedHloWithShapes(hlo, "CHECK: __cudnn$convForward")` is checking a string pattern — it is NOT a cuDNN API call. The test runs on ROCm; it may just fail on the assertion. Always run these.

For clearly justified cases, note them and skip to the next target.

## Step 4: Run each candidate test

For each target that may be unjustified, run it on local GPU (no RBE needed — we just need to know if it runs at all):

```
Tool: Bash
command: .claude/skills/sync-test-check/scripts/run_single_xla_test.sh "<bazel_target>" "" 2>&1 | tail -50
run_in_background: true
```

Wait for each to complete before starting the next — **never run in parallel**.

Use `TaskOutput` (block: true, timeout: 600000).

## Step 5: Handle results

### PASS — tag is unjustified

Remove the `cuda-only` tag from the BUILD file. Commit immediately:

```bash
git add -u
git commit -m "$(cat <<'EOF'
Remove incorrect cuda-only tag from <target>

The cuda-only tag was added upstream but the test passes on ROCm.
<One sentence explaining why the tag is wrong, e.g. "Test self-skips via
GTEST_SKIP() on non-CUDA builds — cuda-only tag is redundant and hides
the test from ROCm CI entirely.">

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

### FAIL — tag is justified

Leave the tag in place. Note it in the final report.

### BUILD_ERROR

The test fails to build — leave the tag, note it as a build issue for `sync-triage-test` to handle later.

## Step 6: Report

**cuda-only tags added upstream:** N
**Skipped (clearly justified):** N
**Tested:** N
  - Tags removed (passes on ROCm): N — list targets
  - Tags kept (genuinely cuda-only): N — list targets
  - Build errors: N — list targets

## Important notes

- Do NOT push anything. The user will push when ready.
- Tests must run sequentially — the `parallel_gpu_execute` flock is non-blocking and causes false failures under contention.
- Each removed tag gets its own commit with the justification in the message.
- If a test passes but you are unsure whether removing the tag is safe, remove it — it can always be re-added. A falsely excluded test is worse than a falsely included one.
