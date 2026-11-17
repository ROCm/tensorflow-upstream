#!/usr/bin/env bash
#
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
#
# Vendored from ROCm/xla (branch rocm-dev-infra),
# build_tools/rocm/detect-therock-version.sh. Kept byte-identical to upstream
# apart from this header so the two stay easy to diff; re-sync rather than
# patching locally.
# =============================================================================
# Script: detect-therock-version.sh
# Purpose: Resolve TheRock (ROCm-via-pip) index URL + version for the TensorFlow
#          Dockerfile.theRock build args, from a release channel + GPU target.
#
# TensorFlow's ci/official/containers/ml_build/Dockerfile.theRock takes two
# build args:
#   THEROCK_INDEX_URL  - the ROCm pip index (GPU-family-specific)
#   THEROCK_VERSION    - the rocm[libraries,devel] version ("" = latest)
#
# These map onto the repo's existing release-channel conventions (the same
# mapping used by scripts/generate-component-config.sh). This script keeps the
# THEROCK_* naming explicit while reusing that proven channel map + detection.
#
# Required environment variables:
#   ROCM_PIP_SOURCE - release channel: nightlies|devreleases|prereleases|release
#   GPU_TARGET      - GPU target for the index path (e.g. gfx950-dcgpu)
#
# Optional environment variables:
#   ROCM_SDK_VERSION    - pin the version (skips auto-detection when set)
#   PIP_INDEX_PATH      - override the index path segment (e.g. whl, whl-staging,
#                         v2, v2-staging); keeps the per-channel domain
#   THEROCK_INDEX_URL   - fully override the index URL (skips channel mapping;
#                         still auto-detects version from this URL unless pinned)
#
# Outputs (written to $GITHUB_OUTPUT if set, always printed):
#   therock_index_url, therock_version
# =============================================================================

set -euo pipefail

ROCM_PIP_SOURCE="${ROCM_PIP_SOURCE:-nightlies}"
GPU_TARGET="${GPU_TARGET:?GPU_TARGET is required}"
ROCM_SDK_VERSION="${ROCM_SDK_VERSION:-}"
PIP_INDEX_PATH="${PIP_INDEX_PATH:-}"
THEROCK_INDEX_URL="${THEROCK_INDEX_URL:-}"

output() {
    local key="$1" val="$2"
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "${key}=${val}" >> "$GITHUB_OUTPUT"
    fi
    echo "${key}=${val}"
}

# --- Construct the index URL ---
# DEVICE_EXTRAS is emitted for the Dockerfile: ",device-all" enables the
# multi-arch kernel packs; empty for single-arch.
DEVICE_EXTRAS=""

# An explicit THEROCK_INDEX_URL override wins outright (channel mapping skipped).
if [ -n "${THEROCK_INDEX_URL}" ]; then
    # Normalize to a single trailing slash.
    THEROCK_INDEX_URL="${THEROCK_INDEX_URL%/}/"
    echo "Using overridden THEROCK_INDEX_URL: ${THEROCK_INDEX_URL}" >&2
    # Heuristic: a multi-arch index implies device-all extras.
    case "${THEROCK_INDEX_URL}" in
        *whl-multi-arch*) DEVICE_EXTRAS=",device-all" ;;
    esac
elif [ -z "${GPU_TARGET}" ] || [ "${GPU_TARGET}" = "device-all" ]; then
    # Multi-arch: per-channel whl-multi-arch index (no gfx segment). The
    # device-all extra pulls per-GPU rocm-sdk-device-* kernel packs.
    case "${ROCM_PIP_SOURCE}" in
        nightlies)   THEROCK_INDEX_URL="https://rocm.nightlies.amd.com/whl-multi-arch/" ;;
        devreleases) THEROCK_INDEX_URL="https://rocm.devreleases.amd.com/whl-multi-arch/" ;;
        prereleases) THEROCK_INDEX_URL="https://rocm.prereleases.amd.com/whl-multi-arch/" ;;
        release)     THEROCK_INDEX_URL="https://repo.amd.com/rocm/whl-multi-arch/" ;;
        *)           echo "ERROR: Unknown ROCM_PIP_SOURCE: ${ROCM_PIP_SOURCE}" >&2; exit 1 ;;
    esac
    DEVICE_EXTRAS=",device-all"
    echo "Multi-arch build: ${THEROCK_INDEX_URL} (device-all)" >&2
elif [ -n "${PIP_INDEX_PATH}" ]; then
    case "${ROCM_PIP_SOURCE}" in
        prereleases) PIP_DOMAIN="rocm.prereleases.amd.com" ;;
        nightlies)   PIP_DOMAIN="rocm.nightlies.amd.com" ;;
        devreleases) PIP_DOMAIN="rocm.devreleases.amd.com" ;;
        *)           PIP_DOMAIN="rocm.prereleases.amd.com" ;;
    esac
    THEROCK_INDEX_URL="https://${PIP_DOMAIN}/${PIP_INDEX_PATH}/${GPU_TARGET}/"
else
    case "${ROCM_PIP_SOURCE}" in
        nightlies)   PIP_BASE="https://rocm.nightlies.amd.com/v2-staging" ;;
        devreleases) PIP_BASE="https://rocm.devreleases.amd.com/v2" ;;
        prereleases) PIP_BASE="https://rocm.prereleases.amd.com/whl-staging" ;;
        release)     PIP_BASE="https://repo.amd.com/rocm/whl" ;;
        *)           echo "ERROR: Unknown ROCM_PIP_SOURCE: ${ROCM_PIP_SOURCE}" >&2; exit 1 ;;
    esac
    THEROCK_INDEX_URL="${PIP_BASE}/${GPU_TARGET}/"
fi

# --- Auto-detect the latest ROCm version unless pinned ---
# pip index HTML is NOT version-sorted, so sort -V and take the last.
#if [ -z "${ROCM_SDK_VERSION}" ]; then
    echo "Detecting latest ROCm version from ${THEROCK_INDEX_URL}rocm-sdk-devel/ ..." >&2
    ROCM_SDK_VERSION=$(curl -sL "${THEROCK_INDEX_URL}rocm-sdk-devel/" \
        | grep -oP 'rocm.sdk.devel-\K[0-9a-z.+]+(?=-py)' | grep "${ROCM_SDK_VERSION}"\
        | sort -V | tail -1 || echo "")
    if [ -z "${ROCM_SDK_VERSION}" ]; then
        echo "ERROR: Could not auto-detect ROCm version from ${THEROCK_INDEX_URL}" >&2
        echo "       Pass ROCM_SDK_VERSION to pin it explicitly." >&2
        exit 1
    fi
    echo "Detected ROCm version: ${ROCM_SDK_VERSION}" >&2
#fi

output "therock_index_url" "${THEROCK_INDEX_URL}"
output "therock_version" "${ROCM_SDK_VERSION}"
output "device_extras" "${DEVICE_EXTRAS}"
