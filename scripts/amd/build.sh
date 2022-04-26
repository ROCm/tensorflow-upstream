# echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
# sudo apt-get update && sudo apt-get install -y openjdk-8-jdk openjdk-8-jre unzip wget git
# cd ~ && rm -rf bazel*.sh && wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh  && bash bazel*.sh && rm -rf ~/*.sh

ROOT_DIR=$(pwd)

(cd "/usr/local/lib/bazel/bin" && curl -fLO https://releases.bazel.build/5.1.1/release/bazel-5.1.1-linux-x86_64 && chmod +x bazel-5.1.1-linux-x86_64)

cd $ROOT_DIR
# bazel clean
pip3 uninstall tensorflow tf-nightly-rocm -y
scripts/amd/build_rocm_python3.sh
pip3 list | grep '^t.*'