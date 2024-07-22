#!/bin/bash

set -e

# Globals.
TF_HOME=$TF_HOME
SCRIPT_DIR="$TF_HOME/tensorflow/core/kernels/CK/"
ARCHIVE_DIR="$SCRIPT_DIR/archives"
DEFAULT_SERVER_PORT=4444
SERVER_URL="http://127.0.0.1"
MIRROR_URL="https://mirror.bazel.build"
CK_ARCHIVE="composable_kernel-internal-letaoqin-develop_base.zip"
GIT_REPO_URL="git@github.com:ROCm/composable_kernel-internal.git"
GIT_REPO_BRANCH="letaoqin/develop_base"
GIT_REPO_DIR="composable_kernel-internal"
BUILD_FILE="//third_party:ck.BUILD"
STRIP_PREFIX="composable_kernel-internal"
WORKSPACE_FILE="$TF_HOME/tensorflow/workspace.bzl"

# Get the current SHA256 from the WORKSPACE file.
get_current_ck_archive_sha256() {
    grep -A 5 'name = "ck_archive"' $WORKSPACE_FILE | grep -oP 'sha256\s*=\s*"\K[^"]+'
}

# Download required packages.
apt install lsof -y

# Find the available port.
# This is currently not used.
find_available_port() {
    local port=$1
    while : ; do
        if ! lsof -i:$port >/dev/null; then
            echo $port
            return
        fi
        port=$((port + 1))
    done
}

# Create the archives directory if it doesn't exist.
mkdir -p $ARCHIVE_DIR
cd $ARCHIVE_DIR

# Clone the repository if it doesn't exist.
if [ ! -d "$GIT_REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone -b $GIT_REPO_BRANCH $GIT_REPO_URL $GIT_REPO_DIR
else
    echo "Repository already cloned. Pulling latest changes..."
    git -C $GIT_REPO_DIR pull
fi

# Zip the repository.
echo "Creating zip archive..."
zip -r $CK_ARCHIVE $GIT_REPO_DIR
cd $SCRIPT_DIR

# Compute the SHA256 checksum of the zip archive.
SHA256=$(sha256sum "$ARCHIVE_DIR/$CK_ARCHIVE" | awk '{print $1}')
echo "New ck archive sha: $SHA256"

# Get the current SHA256 from the WORKSPACE file.
CURRENT_SHA256=$(get_current_ck_archive_sha256)
echo "Current ck archive sha: $CURRENT_SHA256"

# Check if the SHA256 checksum matches the current one in the WORKSPACE file.
if [ "$SHA256" == "$CURRENT_SHA256" ]; then
    echo "Archive is already up to date. Exiting..."
    exit 0
fi

# Find an available port for the local server.
SERVER_PORT=$DEFAULT_SERVER_PORT

# Start the local HTTP server.
echo "Starting local HTTP server on port $SERVER_PORT..."
python3 -m http.server --directory $ARCHIVE_DIR $SERVER_PORT &
# SERVER_PID=$!

# Ensure the server is stopped when the script exits.
# trap "kill $SERVER_PID" EXIT

# Update the existing ck_archive rule in the WORKSPACE file.
echo "Updating ck archive rule..."

# Create a temporary file for the new ck_archive block.
cat <<EOL > /tmp/ck_archive_replacement.txt
tf_http_archive(
        name = "ck_archive",
        build_file = "$BUILD_FILE",
        sha256 = "$SHA256",
        strip_prefix = "$STRIP_PREFIX",
        urls = [
            "$SERVER_URL:$SERVER_PORT/$CK_ARCHIVE",
            "$MIRROR_URL/$CK_ARCHIVE",
        ],
    )
EOL

# Use easier Python approach to replace the workspace.bzl block.
python3 <<EOF
import re

workspace_file = "$WORKSPACE_FILE"
new_block = """$(cat /tmp/ck_archive_replacement.txt)"""

# Read the workspace.bzl file
with open(workspace_file, "r") as file:
    content = file.read()

# Replace the old ck_archive block with the new block
pattern = re.compile(r'tf_http_archive\(\s*name\s*=\s*"ck_archive".*?\)', re.DOTALL)
new_content = pattern.sub(new_block, content)

# Write the updated content back to the workspace.bzl file
with open(workspace_file, "w") as file:
    file.write(new_content)
EOF

# Run Bazel tests.
# echo "Running Bazel tests..."
# bazel test --config=opt --config=rocm --action_env=HIP_PLATFORM=amd --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 //tensorflow/core/kernels/CK/fused_tile_gemm:fused_tile_gemm

# Stop the server.
# kill $SERVER_PID
