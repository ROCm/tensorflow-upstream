---
name: sync
description: "Full end-to-end upstream sync: merge, resolve conflicts, mirror CI, build, and check excluded tests. Orchestrates sync-start → sync-resolve → sync-ci-mirror → sync-build → sync-test-check automatically."
argument-hint: "[YYMMDD (optional, defaults to today)] [optional: upstream ref, defaults to upstream/master]"
---

# Full Upstream Sync — Orchestrator

This skill runs the complete upstream sync pipeline for `ROCm/tensorflow-upstream` end-to-end, without manual intervention. It executes each phase in order by following the instructions in each sub-skill's SKILL.md file.

Arguments are passed through to `/sync-start` (see its SKILL.md for details).

## Pipeline

Execute each phase below **in order**. After each phase completes, proceed immediately to the next one — do not stop to ask the user. Only stop if a phase explicitly says to STOP (e.g., a pre-flight check failure).

### Phase 1: Merge upstream

Follow all instructions in `.claude/skills/sync-start/SKILL.md`.

Pass `$ARGUMENTS` through — they specify the YYMMDD date label and optional upstream ref.

When this phase completes (merge commit recorded), proceed immediately to Phase 2.

### Phase 2: Resolve merge conflicts

Follow all instructions in `.claude/skills/sync-resolve/SKILL.md`.

If Phase 1 reported "Clean merge! No conflicts to resolve", skip this phase entirely.

When this phase completes (conflicts resolved and committed), proceed immediately to Phase 3.

### Phase 3: Mirror upstream CI changes to ROCm counterparts

Follow all instructions in `.claude/skills/sync-ci-mirror/SKILL.md`.

When this phase completes (CI mirroring committed or skipped), proceed immediately to Phase 4.

### Phase 4: Build TensorFlow

Follow all instructions in `.claude/skills/sync-build/SKILL.md`.

When this phase completes (build succeeds, fixes committed if any), proceed immediately to Phase 5.

### Phase 5: Check excluded tests

Follow all instructions in `.claude/skills/sync-test-check/SKILL.md`.

When this phase completes, proceed immediately to Phase 6.

### Phase 6: Validate newly added cuda-only tags

Follow all instructions in `.claude/skills/sync-cuda-only/SKILL.md`.

When this phase completes, proceed immediately to Phase 7.

### Phase 7: Run full XLA test suite on MI250 via RBE

Follow all instructions in `.claude/skills/sync-run-tests/SKILL.md`.

This runs the complete test suite on the EngFlow `linux_x64_gpu_gfx90a` pool and collects all failures.

When this phase completes, record the list of new failing tests (those not already in the exclusion list) and proceed immediately to Phase 8.

### Phase 8: Triage all new failures

For each new failing test identified in Phase 7, follow all instructions in `.claude/skills/sync-triage-test/SKILL.md`.

Process tests **one at a time** (never in parallel). For each test the skill will either fix the code or add it to the exclusion list, committing the result with the full analysis in the commit message.

When all failures have been triaged, proceed immediately to Phase 9.

### Phase 9: Generate test change summary

Produce a markdown summary of all test exclusion changes made during this sync.

**How to extract the data:**

1. Find the merge base between the current branch and `develop-upstream`:
   ```bash
   MERGE_BASE=$(git merge-base HEAD develop-upstream)
   ```

2. Get the diff of the exclusion list files since the merge base:
   ```bash
   git diff $MERGE_BASE HEAD -- tensorflow/tools/ci_build/linux/rocm/run_xla.sh tensorflow/tools/ci_build/linux/rocm/run_gpu_single.sh
   ```

3. Parse the diff to find:
   - **Enabled tests** — lines removed from the exclusion list (diff lines starting with `-` that contain test case patterns or `# @xla//` target comments).
   - **New disabled tests** — lines added to the exclusion list (diff lines starting with `+` that contain test case patterns or `# @xla//` target comments).
   - Ignore diff headers, context lines, section headers (`# vvv TODO`), and blank lines.

4. Group test cases under their target name (from `# @xla//...` comment lines). Separate targets with a blank line.

5. Output the summary in this format:

````markdown
## Technical details

### Enabled tests

```bash
# @xla//xla/path/to:test_target_amdgpu_any
    TestClass.TestCase1
    TestClass.TestCase2
```

### New Disabled Tests

```bash
# @xla//xla/path/to:test_target_amdgpu_any
    TestClass.FailingCase1
    TestClass.FailingCase2

# @xla//xla/path/to:another_test_amdgpu_any
    AnotherClass.FailingCase
```
````

- If no tests were enabled, write "No tests were enabled in this sync."
- If no tests were disabled, write "No new tests were disabled in this sync."

This is the final phase. When complete, print the final summary.

## Final summary

After all phases complete, provide a consolidated summary:

```
## Upstream Sync Complete

**Branch:** develop-upstream-sync-<YYMMDD>
**Upstream ref:** <ref merged>
**Commits on branch:**
1. Merge commit (with unresolved conflicts)
2. Fix merge conflicts
3. Mirror upstream CI changes to ROCm counterparts (if applicable)
4. Fix build errors from upstream sync (if applicable)
5. Remove passing excluded tests (if applicable)
6. Remove incorrect cuda-only tags (if applicable)
7. [Per-test triage commits from Phase 8] (if applicable)

**Conflicts resolved:** N files
**CI files mirrored:** N files (or "none")
**Build fixes:** N files (or "clean build")
**Excluded tests removed:** N (or "none")
**cuda-only tags removed:** N (or "none")
**RBE test results:** X passed, Y failed
**Failures triaged:** N fixed, M excluded

Ready to push and open a PR.
```

## Important notes
- Do NOT push anything. The user will push when ready.
- Do NOT stop between phases to ask the user unless a pre-flight check fails or a phase explicitly requires user input (e.g., ambiguous conflict resolution).
- Each phase's SKILL.md contains the full instructions — read and follow them completely.
- If any phase fails irrecoverably, commit whatever progress has been made and report what failed and what remains.
