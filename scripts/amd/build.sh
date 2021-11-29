# echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
# sudo apt-get update && sudo apt-get install -y openjdk-8-jdk openjdk-8-jre unzip wget git
# cd ~ && rm -rf bazel*.sh && wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh  && bash bazel*.sh && rm -rf ~/*.sh

pip3 uninstall tensorflow -y
./build_rocm_python3