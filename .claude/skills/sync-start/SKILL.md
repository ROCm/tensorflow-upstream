---
name: sync-start
description: Start a regular upstream sync of ROCm tensorflow-upstream with upstream tensorflow. Creates the sync branch, merges upstream, and records the initial merge commit with unresolved conflicts.
argument-hint: "[YYMMDD (optional, defaults to today)] [optional: upstream ref, defaults to upstream/master]"
---

# Upstream Sync Start

You are performing the initial steps of the regular upstream sync for `ROCm/tensorflow-upstream`. This merges changes from upstream `tensorflow/tensorflow` into the `develop-upstream` branch.

Run all steps automatically without asking for confirmation. Report progress and results at the end.

## Arguments

- `$ARGUMENTS[0]` (optional): The YYMMDD date label for this sync (e.g., `260302`). This is the upstream cutoff date, NOT necessarily today's date. **If omitted, defaults to today's date in YYMMDD format** (use the `currentDate` context variable if available, otherwise run `date +%y%m%d` to get it).
- `$ARGUMENTS[1]` (optional): The upstream ref to merge. Defaults to `upstream/master`. Can be a tag like `r2.21-base` or a specific commit.

**Argument disambiguation:** If only one argument is provided, check whether it looks like a YYMMDD date (6 digits). If it does, treat it as the date label and use `upstream/master` as the ref. If it does not look like a date (e.g. it contains letters or slashes), treat it as the upstream ref and derive the date automatically from today.

## Pre-flight checks

Run these checks first and STOP (report the problem) if any fail:

1. Resolve the date label and upstream ref per the argument rules above.
2. Run `git status` — working tree must be clean.
3. Run `git branch --show-current` — must be on `develop-upstream`.
4. Check if branch `develop-upstream-sync-<date-label>` already exists locally or on the remote — must not exist.
5. Check if the `upstream` remote exists (`git remote get-url upstream`). If not, add it:
   ```
   git remote add upstream https://github.com/tensorflow/tensorflow.git
   ```

## Steps

### 1. Fetch upstream
```bash
git fetch upstream
```

### 2. Log what will be merged
Compute and record (for the final summary).
- Total number of commits
```bash
git rev-list --count develop-upstream..upstream/master
```
- The date and hash of the newest (first line) and oldest (last line) commit
```bash
git log --format="%h %ai %s" develop-upstream..<upstream-ref>
```

### 3. Create the sync branch
```bash
git checkout -b develop-upstream-sync-<date-label>
```

### 4. Merge upstream
```bash
git merge <upstream-ref> --no-edit
```

### 5. Handle the merge result

**If merge completes with no conflicts:**
- Show the merge commit stats (files changed, insertions, deletions)
- Print: "Clean merge! No conflicts to resolve. Proceed to building TF."

**If merge has conflicts:**
- Count and list all conflicted files using `git diff --name-only --diff-filter=U`
- Categorize them by area:
  - **XLA/Compiler**: files under `third_party/xla/`
  - **StreamExecutor/GPU runtime**: files under `stream_executor/`
  - **Kernels/Ops**: files under `tensorflow/core/kernels/`
  - **Bazel/Build**: `BUILD`, `.bzl`, `WORKSPACE` files
  - **CI scripts**: files under `tensorflow/tools/ci_build/`
  - **Other**: everything else
- Stage ALL conflicted files as-is (with conflict markers still present):
  ```bash
  git add -u
  ```
  Use `git add -u` (not `git add -A`) — this stages only modifications to tracked files and avoids accidentally including untracked files (e.g. `.claude/`, editor temp files) in the merge commit.
- Finalize the merge commit:
  ```bash
  git commit --no-edit
  ```
  This records the merge with unresolved conflicts as the first commit, which is intentional — it makes PR review easier.
- Report:
  - Total number of conflicted files
  - Categorized list of conflicted files
  - "First commit recorded with unresolved conflicts. Run `/sync-resolve` to resolve them."

## Important notes
- Do NOT attempt to resolve any conflicts. That is the job of `/sync-resolve`.
- Do NOT push anything. The user will push when ready.
- When a specific YYMMDD is given, it represents the upstream cutoff point (e.g. for a tag or a past commit), not necessarily today's date. When syncing against the current `upstream/master` with no date given, today's date is the correct label.
