#!/bin/bash

set -euo pipefail

# Globals.
TF_HOME=${TF_HOME:-"$HOME/tensorflow"}
SCRIPT_DIR="$TF_HOME/tensorflow/core/kernels/CK/"
ARCHIVE_DIR="$SCRIPT_DIR/archives"
GIT_REPO_URL="git@github.com:ROCm/composable_kernel-internal.git"
GIT_REPO_BRANCH="letaoqin/develop_base"
GIT_REPO_DIR="composable_kernel-internal"
BUILD_FILE="//third_party:ck.BUILD"
WORKSPACE_BZL_FILE="$TF_HOME/tensorflow/workspace.bzl"
WORKSPACE_FILE="$TF_HOME/WORKSPACE"

# Download required packages.
if ! command -v lsof &> /dev/null; then
    apt update && apt install lsof -y
fi

# Create the archives directory if it doesn't exist.
mkdir -p "$ARCHIVE_DIR"
cd "$ARCHIVE_DIR"

# Clone the repository if it doesn't exist.
if [ ! -d "$GIT_REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone -b "$GIT_REPO_BRANCH" "$GIT_REPO_URL" "$GIT_REPO_DIR"
else
    echo "Repository already cloned. Pulling latest changes..."
    git -C "$GIT_REPO_DIR" pull
fi

# Remove the existing tf_http_archive rule from the WORKSPACE file.
echo "Removing existing tf_http_archive rule if it exists..."
python3 <<EOF
import re

WORKSPACE_BZL_FILE = "$WORKSPACE_BZL_FILE"

# Read the workspace.bzl file
with open(WORKSPACE_BZL_FILE, "r") as file:
    content = file.read()

# Remove the old tf_http_archive block
pattern = re.compile(r'tf_http_archive\(\s*name\s*=\s*"ck_archive".*?\)', re.DOTALL)
new_content = pattern.sub("", content)

# Write the updated content back to the workspace.bzl file
with open(WORKSPACE_BZL_FILE, "w") as file:
    file.write(new_content)
EOF

# Add the new ck_archive rule to the WORKSPACE file.
echo "Adding new ck_archive rule..."
cat <<EOL >> "$WORKSPACE_FILE"

new_local_repository(
    name = "ck_archive",
    path = "$ARCHIVE_DIR/composable_kernel-internal/",
    build_file = "$BUILD_FILE",
)
EOL

echo "Script completed successfully."
