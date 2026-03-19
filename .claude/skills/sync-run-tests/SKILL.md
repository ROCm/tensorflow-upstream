---
name: sync-run-tests
description: Runs the full XLA test suite on the EngFlow RBE cluster and reports pass/fail results with triage of new failures. Use when user says "run tests", "run RBE tests", "test on RBE", or asks to validate a sync branch against hardware.
---

# Run XLA Tests via RBE

You are running the full XLA test suite against the EngFlow RBE cluster. This uses `.claude/skills/sync-run-tests/scripts/run_xla_rbe.sh`, which builds locally and executes tests remotely on `linux_x64_gpu` workers.

Run all steps automatically without asking for confirmation.

## Pre-flight checks

1. Confirm the current branch matches `develop-upstream-sync-*`. If not, STOP and tell the user.
2. Confirm the TLS certificates required by the RBE cluster exist:
   ```bash
   test -f /tf/certificates/ci-cert.crt && test -f /tf/certificates/ci-cert.key && echo "certs OK"
   ```
   If missing, STOP and tell the user: "RBE TLS certificates not found at /tf/certificates/ci-cert.{crt,key}".
3. Confirm the script exists and is executable:
   ```bash
   test -x .claude/skills/sync-run-tests/scripts/run_xla_rbe.sh && echo "script OK"
   ```
   If not executable, run `chmod +x .claude/skills/sync-run-tests/scripts/run_xla_rbe.sh` and continue.
4. Confirm `ROCM_PATH` is set or `/opt/rocm` exists (needed for the local build):
   ```bash
   echo "ROCM_PATH=${ROCM_PATH:-/opt/rocm}"
   ```

## Step 1: Launch the test run

The build + test run is long (often 1–3 hours). Use background execution:

```
Tool: Bash
command: .claude/skills/sync-run-tests/scripts/run_xla_rbe.sh 2>&1
run_in_background: true
```

Save the returned task ID — you need it to monitor progress.

## Step 2: Monitor progress

After launching, periodically check progress using `TaskOutput` with `block: false`:

```
Tool: TaskOutput
task_id: <task ID from Step 1>
block: false
timeout: 30000
```

- Check every 60–90 seconds
- Report a brief summary to the user each time, e.g.:
  - `"Building: [12,450 / 44,000] actions"`
  - `"Testing: 87 / 439 tests, 3 failed"`
- Key indicators to watch for:
  - `[X / Y]` — Bazel action progress
  - `X / 439 tests` — test execution progress (439 is the total XLA test count)
  - `FAIL:` lines — individual test failures
  - `INFO: Build completed` — run finished
  - `ERROR:` — hard failure

Continue polling until the task status shows completed.

## Step 3: Collect and report results

Once the run completes, get the full output:

```
Tool: TaskOutput
task_id: <task ID from Step 1>
block: true
timeout: 600000
```

Parse the output and report:

### 3.1 Overall result

- Exit code 0 → **PASSED**
- Non-zero → **FAILED**

### 3.2 Build failures

Extract all lines matching `FAILED TO BUILD` — these are test targets that could not compile. List them separately from test failures:

```
⛔ BUILD FAILURES:
- @xla//xla/...:some_test_amdgpu_any — FAILED TO BUILD
```

For each build failure, search the output for the compiler error (look for lines containing `error:` near the target name). Show the error excerpt. Build failures are high priority — they block the test from running at all and usually indicate a merge issue that needs a code fix.

### 3.3 Failed tests

Extract all lines matching `FAIL:` (but NOT `FAILED TO BUILD`) and list them as a table:

| Test target | Log path |
|-------------|----------|
| `@@xla//xla/...` | `/path/to/test.log` |

For each failure, read the test log (the path is printed after `FAIL:`) and extract the first error or assertion failure. Show a brief excerpt.

### 3.4 Summary table

```
Total tests:   441
Passed:        X
Failed:        Y
Build failures: Z
(Not run):     W
```

## Step 4: Handle failures

For each failed test:

1. Read its test log to determine the failure type:
   - **Known ROCm limitation** (e.g. unsupported op, HIP error): note it, do not add to exclusion list yet — report to user for triage.
   - **glibc / linker error**: this is an environment issue, not a test failure — report it separately.
   - **Assertion / correctness failure**: likely a real failure — report for triage.

2. Do NOT automatically add tests to the exclusion list. That is the user's decision.

3. **Cross-reference at the test CASE level, not the target level.** The RBE run does not apply `--test_filter`, so a test binary that has some cases excluded may still fail due to NEW cases not yet in the exclusion list. For each failed test target:
   a. Read the test log and extract all `[  FAILED  ]` lines to get the specific failing test case names.
   b. Check each failing case against the `EXCLUDED_TESTS` entries in `run_xla.sh`.
   c. A test target is only "fully excluded" if **every** failing case from that target matches an entry in `EXCLUDED_TESTS`. If even one failing case is not covered, report it as a new failure.

4. If ALL failing cases across all targets are pre-existing (already in `EXCLUDED_TESTS`), report: "All failures are already in the exclusion list — no new failures."

5. If there are NEW failures (not in the exclusion list), highlight them clearly with the specific case names:
   ```
   ⚠ NEW FAILURES (not in exclusion list):
   - @xla//xla/...:some_test_amdgpu_any — SomeTestClass.SomeTestCase
   ```

## Important notes

- Do NOT push anything. The user will push when ready.
- Do NOT modify any source files or exclusion lists during this skill — only observe and report.
- The EngFlow BES URL is printed at the start of the bazel run. Capture and report it so the user can inspect the run in the web UI.
- If the run is interrupted (e.g. network error, certificate expiry), report the last known progress and suggest re-running the skill.
- Queue wait times may be long if the pool is busy. Monitor the output for `(Sched)` annotations indicating queued tests.
