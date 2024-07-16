#!/usr/bin/env bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
#
# setup.python.sh: Install a specific Python version and packages for it.
# Usage: setup.python.sh <pyversion> <requirements.txt>
set -xe

source ~/.bashrc

apt-get update
# Install python dep
apt-get install -y python-dev
# Install bz2 dep
apt-get install -y libbz2-dev
# Install curses dep
apt-get install -y libncurses5 libncurses5-dev
apt-get install - libncursesw5 libncursesw5-dev
# Install readline dep
apt-get install -y libreadline8 libreadline-dev
# Install sqlite3 dependencies
apt-get install -y libsqlite3-dev

# PYTHON
wget https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tgz && tar xvf Python-3.7.17.tgz && cd Python-3.7*/ && ./configure --enable-optimizations && make altinstall
ln -sf /usr/local/bin/python3.7 /usr/bin/python3 && ln -sf /usr/local/bin/pip3.7 /usr/bin/pip3
ln -sf /usr/local/bin/python3.7 /usr/bin/python && ln -sf /usr/local/bin/pip3.7 /usr/bin/pip
ln -sf /usr/local/lib/python3.7 /usr/lib/tf_python

NUMPY_VERSION=1.14.5

export PYTHON_LIB_PATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages
export PYTHON_BIN_PATH=/usr/local/bin/python${PYTHON_VERSION}

# Install pip and venv
curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py
python3 get-pip.py
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --upgrade virtualenv
