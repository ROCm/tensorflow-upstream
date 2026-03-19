---
name: sync-resolve
description: Resolve merge conflicts after /sync-start. Analyzes the cause of each conflict using git history, resolves using ROCm conventions, and commits the fix.
---

# Resolve Upstream Sync Conflicts

You are resolving merge conflicts created during the upstream sync of `ROCm/tensorflow-upstream`. The `/sync-start` skill has already created the sync branch and made the first commit with unresolved conflicts. In this context:
- **Ours** = ROCm (`develop-upstream`) — our changes and ROCm-specific additions
- **Theirs** = upstream (`tensorflow/tensorflow`) — the new upstream changes being merged in

Reference the full sync process in `sync-process.md` at the repo root.

Run all steps automatically. Only pause to ask the user when a conflict is genuinely ambiguous or a large conflict (>100 lines) needs confirmation of approach.

## Pre-flight checks

1. Confirm the current branch matches `develop-upstream-sync-*`. If not, STOP.
2. Confirm the working tree is clean. If not, STOP.

## Step 1: Find all conflict markers

```bash
grep -rn "<<<<<<" --include="*" -l
```

If none found: report "No conflict markers found." and STOP.

List all conflicted files categorized by area:
- **XLA/Compiler**: `third_party/xla/`
- **StreamExecutor/GPU runtime**: `stream_executor/`
- **Kernels/Ops**: `tensorflow/core/kernels/`
- **Bazel/Build**: `BUILD`, `.bzl`, `WORKSPACE`
- **CI scripts**: `tensorflow/tools/ci_build/`
- **Other**

## Step 2: Resolve conflicts file by file

Process in this order: Bazel/Build → XLA → StreamExecutor → Kernels → CI → Other.

For EACH conflicted file:

### 2.1 Analyze the conflict and its cause (CRITICAL)

Read the conflicted file to understand the conflict markers:
- `<<<<<<< HEAD` — our version (ROCm/develop-upstream)
- `=======` — separator
- `>>>>>>> <commit>` — their version (upstream)

Then **investigate WHY the conflict occurred** using git history on **both branches independently**.

#### Step A: Identify the semantic entity containing the conflict

Do not track raw line numbers — they differ between branches. Instead, identify the enclosing semantic entity (Bazel target name, function name, class name) by reading the file around the conflict markers. Then find the line range of that entity on each branch separately:

```bash
# Find the entity's line range on our branch
git show develop-upstream:<file> | grep -n '<entity_name>'

# Find the entity's line range on upstream
git show upstream/master:<file> | grep -n '<entity_name>'
```

Read a window around each result to confirm the start and end lines of the full entity on each branch.

#### Step B: Run git log -L on each branch with its own line numbers

```bash
# Full history of the entity on our branch (use line numbers from develop-upstream)
git log -L <our_start>,<our_end>:<file> develop-upstream

# Full history of the entity on upstream (use line numbers from upstream/master)
git log -L <upstream_start>,<upstream_end>:<file> upstream/master
```

This reveals the complete evolution of the conflicting region on each side — every commit that touched it, in chronological order.

#### Step C: Look at the broader context of each side's history

When reading the `git log -L` output, pay attention to:

- **Intentional upstream deletions**: If upstream removed something (a tag, a parameter, a workaround), find the commit that removed it and read its message. If it says the underlying issue was resolved (e.g., "Enable test after CUDA driver update", "Remove deprecated flag"), that deletion is intentional and correct — do NOT restore it from our side.
- **ROCm code introduced by a previous sync merge**: Run `git log -L` on `develop-upstream` and check whether the conflicting ROCm-side lines were introduced in a prior sync merge commit (message like "Fix merge conflicts" or "Merge remote-tracking branch 'upstream/master'"). If so, that code may itself be a mistake from a prior sync resolution, not original ROCm work. Treat it with suspicion.
- **TODOs and workarounds that reference resolved conditions**: If the HEAD side contains a comment like `# TODO: Re-enable once X is fixed` and upstream removed it, check whether upstream removed it *because X was fixed*. If yes, upstream's removal wins.

#### Step D: Document your findings

Before choosing a resolution strategy, record for each conflict:
- Which commit(s) on `develop-upstream` introduced the conflicting region, and what they did — **include the short commit hash**
- Which commit(s) on `upstream/master` last touched the same region, and what they did — **include the short commit hash**
- Whether upstream's change is additive, a deletion of resolved code, or a refactor
- Whether our side's content is original ROCm work or was re-introduced by a prior sync

Capture the upstream commit hash with:
```bash
git log --oneline -1 upstream/master -- <file>
# or, when git log -L was used, read the hash from its output
```

This analysis drives the resolution decision **and feeds directly into the commit message** (Step 4).

### 2.2 Choose a resolution strategy

**Strategy A: Independent changes — MERGE BOTH**
If ours and theirs modified different parts or are complementary (most common for ROCm additions alongside upstream changes):
- Keep both changes in logical order (upstream first, ROCm additions after)

**Strategy B: Prefer upstream — TAKE THEIRS**
If upstream's version is a refactoring, API migration, or improvement that our side lacks:
- Take upstream's version as the base, then re-apply any ROCm-specific additions on top

**Strategy C: Prefer ours — TAKE OURS**
If our side already has a more complete implementation that supersedes the upstream change:
- Keep our version. Common when ROCm already generalized a CUDA-only change.
- **Do NOT use this strategy** if the ROCm-side content was introduced by a prior sync merge conflict resolution (rather than deliberate ROCm work), or if upstream's version deleted a resolved workaround. In those cases use Strategy B.

**Strategy D: Semantic merge — COMBINE INTELLIGENTLY**
If both sides made meaningful changes to the same code:
- Understand what each was trying to accomplish
- Create a merged version that preserves both intents
- Ensure result is syntactically and semantically correct

### 2.2.1 Function signature conflicts — check for additive resolution

When a conflict involves **function parameters** where each side has a *different* parameter (not the same parameter modified), this is almost always an **additive** situation requiring both parameters. Do NOT treat it as "ours vs theirs".

**Before choosing a resolution:**
1. **Check usage in the function body.** Search for each conflicting parameter name in the rest of the function. If both are referenced, you MUST keep both.
2. **Check callers.** Find all call sites of the function. If callers pass the ROCm-side parameter, it cannot be dropped.
3. **Check git history.** Determine if the upstream parameter is *new* (added in the merge commit) vs the ROCm parameter is *existing* (present before the merge). If one is new and one is existing, you almost certainly need both.

**Example of this mistake:**
```
<<<<<<< HEAD
    const DebugOptions& debug_options,
    llvm_ir::LLVMCommandLineOptionsLock& llvm_lock) {
=======
    const DebugOptions& debug_options, bool keep_tempfiles) {
>>>>>>> upstream/master
```
Wrong: picking one side. Correct: keeping both (`keep_tempfiles` AND `llvm_lock`), since `keep_tempfiles` was new upstream and `llvm_lock` was existing ROCm code used later in the function.

### 2.2.2 Verify no ROCm code is orphaned by the resolution

After resolving any conflict, **grep the rest of the file** for identifiers that existed on the ROCm side of the conflict but were removed in the resolution. If any removed identifier is still referenced elsewhere in the file, the resolution is wrong — you dropped something that's still needed.

```bash
# For each identifier removed from the ROCm side, check if it's used elsewhere
grep -n "<removed_identifier>" <file>
```

### 2.3 ROCm-specific resolution rules

Apply these on top of the strategy chosen above:

#### Bazel / BUILD files
- Keep both CUDA and ROCm dependencies
- If upstream renamed `cuda_` → `gpu_`, accept the rename but preserve ROCm equivalents
- When upstream moved deps, follow the move and keep ROCm-specific deps (`rocm`, `TENSORFLOW_WITH_ROCM`, `if_rocm`)
- Default: upstream version as base + ROCm additions on top

**`cuda-only` tag handling (important):**
Only use `tags = ["cuda-only"]` for tests that are genuinely CUDA-only by design (i.e. they test a CUDA-specific feature and cannot run on ROCm). Do NOT use it to disable failing ROCm tests — that practice caused tests to be forgotten for years.

When you encounter `tags = ["cuda-only"]` during conflict resolution or in an upstream file being merged:
1. **Read the test source file.** Look for:
   - `GTEST_SKIP()` with a ROCm-related message — the test already self-skips on ROCm; `cuda-only` is redundant and hides it from ROCm CI entirely
   - `is_built_with_rocm_` guards — same conclusion
   - `cuda_compute_capability()` used only in `if/else` branches to select expected output patterns — the test still **runs** on ROCm via the `else` branch; `cuda-only` is wrong
   - `cuda_compute_capability()` used as a hard gate (e.g. `if (!cuda_cap) return;`), or the test exercises cuDNN/NCCL/cuBLAS internals exclusively — genuinely CUDA-only
2. **If source inspection shows it is not truly CUDA-only**: remove the `cuda-only` tag. The test will be run and verified in the `sync-test-check` phase after the build; if it fails there it will be added to the exclusion list. Upstream may still carry this tag — that is expected; we simply do not propagate it on our branch.
3. **If source inspection confirms it is truly CUDA-only**: keep the tag.

#### XLA (`third_party/xla/`)
- Accept upstream API signature changes
- Preserve ROCm-specific code blocks (`#if TENSORFLOW_USE_ROCM` or `ROCM` guards)
- For `#if GOOGLE_CUDA` / `#elif TENSORFLOW_USE_ROCM` pairs: keep both, update the ROCm block to match any API changes in the CUDA block

**`XlaSrcRoot()` path fix (critical — recurring pattern):**
XLA is vendored into TensorFlow via `third_party/xla/`, which means test data files end up under `external/xla/xla/` in bazel runfiles rather than directly under `xla/`. Many XLA test files have an ROCm-side path fix like:
```cpp
auto path = tsl::testing::XlaSrcRoot();
path = path.erase(path.length() - 4);
// then use: tsl::io::JoinPath(path, "external/xla/xla", ...)
```
This pattern exists in multiple test files (e.g. `amdgpu_register_spilling_test.cc`, `xla_gpu_compile_lib_test.cc`, `xla_deviceless_compile_lib_test.cc`). When a conflict involves code that uses `tsl::testing::XlaSrcRoot()`, **always preserve this path adjustment from the ROCm side**. Upstream's direct `XlaSrcRoot()` path works in the upstream XLA repo but breaks in the vendored tensorflow-upstream build.

#### StreamExecutor / GPU runtime
- Accept upstream interface changes
- Preserve ROCm implementations alongside CUDA ones
- Mirror the CUDA version's pattern for the ROCm version

#### Kernels/Ops (`tensorflow/core/kernels/`)
- If upstream added `#if GOOGLE_CUDA` without ROCm coverage, check if the op should work on ROCm — if so, change to `#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM`
- Preserve existing `TENSORFLOW_USE_ROCM` guards

#### CI scripts (`tensorflow/tools/ci_build/`)
- Keep ROCm-specific CI scripts and test configurations
- Accept upstream changes to shared infrastructure

#### Imports / includes
- Keep all unique imports from both sides, remove duplicates
- Sort per project conventions

#### General
- **Never leave `<<<<<<<`, `=======`, or `>>>>>>>` markers in any file**
- When upstream deleted code that had ROCm additions: deletion usually wins unless the ROCm code is independent
- When upstream added new code in the same location as ROCm code: keep both, upstream first
- **Cross-file consistency:** When the same conflict pattern appears in multiple files (e.g. the same path fix or the same function signature change), resolve one carefully, then apply the same resolution to all others. Do not resolve each file independently — this leads to inconsistent resolutions where some files get the fix and others don't.

### 2.4 Apply the resolution

Use the Edit tool to replace each conflict block (including markers) with the resolved code.

### 2.5 Run tests for resolved test files

If a conflicted file is a test source (e.g. `*_test.cc`), **run that test locally** after resolving the conflict to verify the resolution is correct. A test that crashes or fails immediately after conflict resolution is a strong signal that something was dropped or merged incorrectly. This catches issues like broken paths, missing parameters, or wrong function signatures before they reach CI.

Use the helper scripts from sync-test-check if available, or run via bazel directly:
```bash
bazel test --config=rocm <test_target> 2>&1 | tail -20
```

Do not block on long-running tests — focus on tests that are quick to build and run (< 2 minutes). For slow tests, note them in the report as needing verification.

## Step 3: Verify no markers remain

```bash
grep -rn "<<<<<<" --include="*" -l
```

If any remain, go back and resolve them.

Also spot-check for stray `=======` and `>>>>>>>` — these can have false positives (test data, docs), so review each match:

```bash
grep -rn "=======" --include="*" | grep -v "Binary file" | head -20
grep -rn ">>>>>>>" --include="*" | grep -v "Binary file" | head -20
```

## Step 4: Commit the resolution

The commit message must include the full reasoning for every conflict — not just "Fix merge conflicts". Use this format:

```
Fix merge conflicts from upstream sync

<file1> (upstream <hash>: <one-line description of upstream change>)
  ROCm: <what our side had>
  Decision: <strategy + why, e.g. "Took upstream — py_strict migration, python_version deprecated">

<file2> (upstream <hash>: <one-line description of upstream change>)
  ROCm: <what our side had>
  Decision: <strategy + why>

... one block per conflicted file ...
```

Rules:
- The upstream commit hash must be the actual short hash from `git log` — not a placeholder.
- If multiple files share the same upstream cause (e.g. the same migration commit touched 4 BUILD files), group them under one header with that shared hash, then list each file's decision.
- If a ROCm-side change was introduced by a prior sync merge (not deliberate ROCm work), say so explicitly.

Example:
```
Fix merge conflicts from upstream sync

tensorflow/dtensor/python/tests/BUILD (upstream 3b3ee72: py_strict → py_test migration)
tensorflow/python/autograph/utils/BUILD (upstream 3b3ee72)
tensorflow/python/debug/wrappers/BUILD (upstream 3b3ee72)
  ROCm: python_version = "PY3", srcs_version = "PY3"
  Decision: Took upstream — python_version deprecated, replaced with strict_deps = True.
            debug/wrappers: also preserved ROCm's tags=[] (no-rocm tag intentionally cleared).

tensorflow/lite/python/BUILD (upstream 3b3ee72: py_strict migration + tf_python_pybind_extension removal)
  ROCm: old load statements including tf_python_pybind_extension
  Decision: Took upstream load statements; no ROCm-specific additions in that section.

third_party/xla/xla/service/gpu/tests/BUILD (upstream a399d9a: cuda-only tag added to gpu_too_many_blocks_test)
  ROCm: tags = ["pjrt_migration_candidate"]
  Decision: Removed cuda-only — source shows test self-skips on ROCm via is_built_with_rocm_; tag would hide it from ROCm CI.
```

Commit with:
```bash
git add -A
git commit -F - <<'EOF'
Fix merge conflicts from upstream sync

... message body ...
EOF
```

## Step 5: Report

Provide a structured summary:

### Per-file conflict analysis

For each resolved file:

```
### `path/to/file`

**Cause:** What changed on upstream (commit `abc123` by Author, Date): brief description.
What changed on HEAD that conflicted: brief description.

**Resolution:** Strategy used — brief explanation of what was done.
```

### Summary table

| File | Strategy | Reason |
|------|----------|--------|
| file1.cc | Merge both | ROCm guard preserved alongside upstream API change |
| BUILD | Prefer upstream | Dep rename accepted, ROCm deps re-added |

### Concerns / items needing manual verification
- List any ambiguous resolutions or semantic conflicts that couldn't be automatically verified
- Note anything that may need a build check

Finish with: "Build TF to verify the merge compiles. Fix any build errors in follow-up commits."

## Important notes
- Do NOT try to fix build errors. That is a separate step. However, if the build-fix phase requires adding a **new workaround** (e.g., a static mutex, a reimplemented helper) to replace functionality that existed in the pre-merge ROCm code, this is a strong signal that a conflict was resolved incorrectly. Go back and check if the original ROCm code was accidentally dropped during conflict resolution.
- Do NOT modify test exclusion scripts (`run_xla.sh`, `run_gpu_single.sh`). That happens after CI runs.
- Do NOT push anything.
- For very large conflicts (>100 lines), summarize your planned resolution and confirm with the user before editing.
- If a conflict is genuinely ambiguous after git history analysis, show the user both sides and your recommendation before resolving.
