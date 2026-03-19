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
# Usage: tag-post.sh YYMMDD
# Step 3 of 3 in the post-approval workflow (run AFTER merging the PR).
# Checks out develop-upstream, pulls the merged result, and tags the new HEAD
# as merge-YYMMDD to mark the completed sync point.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 YYMMDD" >&2
    exit 1
fi

YYMMDD="$1"

git checkout develop-upstream
git fetch origin
git pull origin develop-upstream
git tag "merge-${YYMMDD}"
git push origin "merge-${YYMMDD}"
