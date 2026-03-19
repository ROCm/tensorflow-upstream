#!/usr/bin/env bash
# ==============================================================================
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
# ==============================================================================
# Usage: tag-prev.sh YYMMDD
# Step 1 of 3 in the post-approval workflow (run BEFORE merging the PR).
# Checks out develop-upstream, pulls, and tags the current HEAD as
# merge-YYMMDD-prev so there is a clear baseline of what existed before the merge.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 YYMMDD" >&2
    exit 1
fi

YYMMDD="$1"

git checkout develop-upstream
git fetch origin
git pull origin develop-upstream
git tag "merge-${YYMMDD}-prev"
git push origin "merge-${YYMMDD}-prev"
