---
name: sync-build
description: Build TensorFlow after sync conflict resolution. Runs build_rocm_python3, diagnoses and fixes any build failures, and repeats until the build succeeds and a wheel is produced.
---

# Build TensorFlow After Upstream Sync

You are building TensorFlow after the upstream sync conflict resolution for `ROCm/tensorflow-upstream`. The `/sync-start` and `/sync-resolve` skills have already completed — the sync branch has a first commit with unresolved conflicts and a second commit that fixes them. Now you need to verify the merge compiles and produces a working wheel.

**Note:** Run `/sync-ci-mirror` before this skill to check for upstream CI changes that need ROCm mirroring.

Run all steps automatically without asking for confirmation unless a fix is genuinely ambiguous.

## Pre-flight checks

Run these checks first and STOP (report the problem) if any fail:

1. Confirm the current branch matches `develop-upstream-sync-*`. If not, STOP.
2. Confirm there are no remaining conflict markers:
   ```bash
   grep -rn "<<<<<<" --include="*" -l
   ```
   If any are found, STOP and tell the user to run `/sync-resolve` first.
3. Confirm `build_rocm_python3` exists in the repo root and is executable.

## Step 1: Run the build

The TF build takes a very long time (often 1-3 hours). You MUST use background execution with periodic progress monitoring.

### 1.1 Launch the build in the background

Use the Bash tool with `run_in_background: true` to start the build. This removes the timeout limit and lets you monitor progress:

```
Tool: Bash
command: ./build_rocm_python3 2>&1
run_in_background: true
```

This returns a **task ID**. Save it — you need it to monitor progress.

### 1.2 Monitor build progress

After launching, periodically check progress using the `TaskOutput` tool with `block: false` to get a non-blocking snapshot of the current output:

```
Tool: TaskOutput
task_id: <the task ID from 1.1>
block: false
timeout: 30000
```

- Report a brief progress summary to the user each time (e.g., "Bazel is compiling... 1,523 of 24,000 actions")
- Check every 60-90 seconds using `TaskOutput` with `block: false`
- Look for key progress indicators in the output:
  - `Analyzing:` — Bazel is loading/analyzing the build graph
  - `[X,XXX / YY,YYY]` — Bazel action progress (X of Y actions completed)
  - `Building ...` — compilation in progress
  - `PASSED` / `FAILED` — build finished
  - `ERROR` — build error occurred
- Continue polling until the task status shows completed (either success or failure)

### 1.3 Check build result

Once the task completes, check the final output for the exit code.

**Success criteria:** The build is successful if and only if:
- The `build_rocm_python3` script exits with code 0, AND
- A `.whl` file exists in `bazel-bin/tensorflow/tools/pip_package/wheel_house/`

Verify the wheel exists:
```bash
ls bazel-bin/tensorflow/tools/pip_package/wheel_house/*.whl
```

If the build succeeds on the first try, skip to **Step 3: Commit and report**.

## Step 2: Diagnose and fix build failures (iterate)

If the build fails, follow this loop:

### 2.1 Analyze the build error

Read the build output carefully. Common failure patterns after an upstream sync:

- **Missing/renamed files:** Upstream moved or deleted files that ROCm code references. Fix imports/includes and BUILD deps.
- **API changes:** Upstream changed function signatures, class names, or namespaces. Update ROCm code to match the new API.
- **Duplicate symbols:** Both sides added similar code. Remove the duplicate, keeping the upstream version and preserving ROCm-specific behavior.
- **Missing BUILD dependencies:** New upstream code needs deps that aren't in ROCm BUILD files. Add the missing deps.
- **Incompatible macro/config changes:** Bazel macros or config settings changed upstream. Adapt ROCm build configs accordingly.
- **Header include path changes:** Upstream reorganized headers. Update `#include` paths.

### 2.2 Fix the error

Apply the minimal fix needed to resolve the build error. Follow these principles:

- **Minimal changes:** Fix only what is broken. Do not refactor or clean up surrounding code.
- **Follow upstream patterns:** When adapting to upstream API changes, follow the same patterns used in upstream's own code.
- **Preserve ROCm code:** Keep `#if TENSORFLOW_USE_ROCM` blocks, ROCm-specific implementations, and ROCm BUILD deps. Adapt them to new APIs rather than removing them.
- **Mirror CUDA patterns:** If upstream changed CUDA code, make the corresponding change in the ROCm equivalent.
- **Do NOT modify test exclusion scripts** (`run_xla.sh`, `run_gpu_single.sh`). That happens after CI runs.

### 2.3 Rebuild

After applying fixes, run the build again using the same background pattern from Step 1:

```
Tool: Bash
command: ./build_rocm_python3 2>&1
run_in_background: true
```

Monitor progress with `TaskOutput` (block: false) as before. Then verify the success criteria again (exit code 0 + wheel file exists).

### 2.4 Repeat

Continue the diagnose → fix → rebuild cycle until the build succeeds. If after 5 iterations the build still fails:
- Commit whatever fixes have been applied so far
- Report the remaining build error in detail
- STOP and ask the user for guidance

## Step 3: Commit and report

Once the build succeeds:

### 3.1 Commit build fixes (if any)

If any code changes were made to fix build errors:

```bash
git add -u
git commit -m "Fix build errors from upstream sync"
```

Use `git add -u` (not `git add -A`) to avoid accidentally staging untracked files.

If no code changes were needed (build passed on first try), skip the commit.

### 3.2 Report

Provide a structured summary:

**Build result:** SUCCESS

**Wheel location:**
```
bazel-bin/tensorflow/tools/pip_package/wheel_house/<wheel-filename>.whl
```

**Build fixes applied** (if any):

For each file modified:
```
### `path/to/file`
**Error:** Brief description of the build error
**Fix:** What was changed and why
```

**Summary table** (if fixes were applied):

| File | Error Type | Fix |
|------|-----------|-----|
| file1.cc | Missing include | Updated include path to match upstream rename |
| BUILD | Missing dep | Added new upstream dep |

**If no fixes were needed:**
"Build passed cleanly after conflict resolution. No additional changes required."

Finish with: "Upstream sync build verified. Ready to push and open a PR."

## Important notes
- Do NOT push anything. The user will push when ready.
- Do NOT modify test exclusion scripts.
- The build can take a very long time. Be patient and let it complete.
- When fixing build errors, prefer the smallest possible change. Each fix should address exactly one error.
- If the same file needs multiple fixes across rebuild iterations, that is fine — they will all be captured in a single "Fix build errors" commit.
