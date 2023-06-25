#!/bin/bash

git clone https://github.com/d1egoprog/tensorflow-gpu-jupyter-docker.git

cd tensorflow-gpu-jupyter-docker

cp ../requirements.txt .
cp ../Dockerfile_tensorflow Dockerfile

docker build -t myjupyterlab-p7 .

