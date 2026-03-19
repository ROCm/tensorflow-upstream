---
name: sync-ci-mirror
description: Mirror upstream CI changes to ROCm counterparts. Checks ci/official and tf_sig_build_dockerfiles for upstream changes and applies matching updates to ROCm-specific Dockerfiles, env configs, and package lists.
---

# Mirror Upstream CI Changes to ROCm Counterparts

You are checking whether the upstream sync introduced CI changes that need to be mirrored into ROCm-specific files. This is part of the regular upstream sync for `ROCm/tensorflow-upstream`.

Upstream CI files have ROCm-specific counterparts that must stay in sync. When upstream modifies a base file, the corresponding ROCm file may need matching updates.

Run all steps automatically without asking for confirmation unless a change is genuinely ambiguous.

## Pre-flight checks

1. Confirm the current branch matches `develop-upstream-sync-*`. If not, STOP.
2. Confirm the working tree is clean (`git status`). If not, STOP.

## Step 1: Identify what upstream changed

Check if the upstream merge touched any of the CI base files:

```bash
git diff develop-upstream...HEAD --name-only -- ci/official/ tensorflow/tools/tf_sig_build_dockerfiles/
```

If no CI files were changed, report "No upstream CI changes require ROCm mirroring." and STOP.

## Step 2: Analyze changes against known upstream → ROCm file pairs

For each changed file, check if it has a ROCm counterpart that needs updating. The known pairs are listed below.

### `ci/official/containers/ml_build/` (new official CI)

| Upstream file | ROCm counterpart |
|---|---|
| `Dockerfile` | `Dockerfile.rocm` |
| `setup.packages.sh` | *(shared — used by both)* |
| `setup.python.sh` | *(shared — used by both)* |
| `setup.sources.sh` | *(shared — used by both)* |
| `builder.packages.txt` | *(shared — used by both)* |
| `builder.requirements.txt` | *(shared — used by both)* |

The ROCm Dockerfile (`Dockerfile.rocm`) mirrors the upstream `Dockerfile` but replaces CUDA/NVIDIA sections with ROCm installation. When upstream changes the `Dockerfile`, check whether the same change applies to `Dockerfile.rocm` — specifically:
- Python version additions/removals/reordering
- Tool version bumps (bats, bazelisk, buildifier, buildozer, patchelf)
- New shared setup steps or COPY instructions
- Base image updates
- Bazel cache configuration changes

The shared scripts (`setup.packages.sh`, `setup.python.sh`, etc.) are used by both Dockerfiles — if upstream changed these, the ROCm build benefits automatically. No ROCm mirroring needed for those.

### `ci/official/envs/`

| Upstream file | ROCm counterpart |
|---|---|
| `linux_x86` | `linux_x86_rocm` |

The ROCm env file sources (`source ci/official/envs/linux_x86`) the upstream base and overrides specific variables. If upstream changed `linux_x86`, check whether:
- New variables were added that `linux_x86_rocm` should override
- Existing variables that `linux_x86_rocm` overrides were renamed or removed
- Default values changed that affect ROCm builds

### `tensorflow/tools/tf_sig_build_dockerfiles/` (SIG build)

| Upstream file | ROCm counterparts |
|---|---|
| `Dockerfile` | `Dockerfile.rocm` (primary), `Dockerfile.rocm.ub20`, `Dockerfile.rocm.ub22`, `Dockerfile.rocm.ub24`, `Dockerfile.rocm.manylinux_2_28` |
| `setup.packages.sh` | `setup.packages.rocm.el8.sh`, `setup.packages.rocm.cs7.sh` (distro-specific) |
| `devel.packages.txt` | `devel.packages.rocm.txt`, `devel.packages.rocm.el8.txt`, `devel.packages.rocm.cs7.txt` |
| `builder.packages.txt` | `builder.packages.rocm.el8.txt`, `builder.packages.rocm.cs7.txt` |
| `setup.sources.sh` | *(shared — used by both)* |
| `devel.requirements.txt` | *(shared — used by both)* |

The SIG build ROCm Dockerfiles are more divergent from upstream (different base images, multi-stage vs single-stage), but shared sections should still be mirrored:
- Tool versions (bats, bazelisk, buildifier, buildozer, patchelf)
- Python setup changes
- Package list updates (new shared dependencies)
- Build environment configuration

## Step 3: Apply mirroring updates

For each upstream file that was changed in this sync and has a ROCm counterpart:

1. Read the upstream file's diff to understand what changed:
   ```bash
   git diff develop-upstream...HEAD -- <upstream-file>
   ```
2. Read the ROCm counterpart file
3. Determine if the change applies to the ROCm version:
   - **YES, mirror it:** Tool version bumps, Python changes, shared infrastructure updates, new build steps
   - **NO, skip it:** CUDA/NVIDIA-specific changes, changes already handled differently in the ROCm file
4. Apply the change to the ROCm file using the Edit tool

## Step 4: Commit

If any ROCm CI files were updated:

```bash
git add -u
git commit -m "Mirror upstream CI changes to ROCm counterparts"
```

Use `git add -u` (not `git add -A`) to avoid accidentally staging untracked files.

If no mirroring was needed (all upstream changes were to shared files or NVIDIA-specific), skip the commit.

## Step 5: Report

Provide a structured summary:

### Upstream CI files changed
- List all CI files modified by the upstream merge

### ROCm counterparts updated

For each file updated:
```
### `path/to/rocm-file`
**Upstream change:** What changed in the upstream file
**Mirrored:** What was updated in the ROCm counterpart and why
```

### Changes skipped
- List upstream changes that did NOT require ROCm mirroring and why (e.g., "NVIDIA-specific", "shared file — ROCm benefits automatically")

### Summary table

| Upstream file | ROCm counterpart | Action | Reason |
|---|---|---|---|
| Dockerfile | Dockerfile.rocm | Updated | Python 3.14 added |
| setup.python.sh | *(shared)* | Skipped | Shared file, no mirroring needed |

## Important notes
- Do NOT push anything. The user will push when ready.
- Do NOT modify non-CI files. This skill is strictly for CI/official and tf_sig_build_dockerfiles mirroring.
- When in doubt about whether a change applies to ROCm, mirror it — it is safer to keep the ROCm files up to date than to miss an update.
