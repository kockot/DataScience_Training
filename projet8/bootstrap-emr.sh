#!/bin/bash

sudo yum update -y
sudo yum install -y automake fuse fuse-devel gcc-c++ git libcurl-devel libxml2-devel make openssl-devel

git clone https://github.com/s3fs-fuse/s3fs-fuse.git
cd s3fs-fuse
./autogen.sh
./configure --prefix=/usr --with-openssl
make
sudo make install
echo "AKIAYQ42R3GTM336ODN4:o59H7ltG8nWEqQUB/T0OpFTh1nw1mpJwQbZYRrwG" | sudo tee /etc/passwd-s3fs
sudo chmod 640 /etc/passwd-s3fs

sudo mkdir /kockot-bucket
sudo s3fs kockot-bucket -o use_cache=/tmp -o allow_other -o uid=1001 -o umask=000 -o multireq_max=5 /kockot-bucket


sudo python3 -m pip install -U setuptools
sudo python3 -m pip install pip
sudo python3 -m pip install wheel
sudo python3 -m pip install pillow
sudo python3 -m pip install pandas
sudo python3 -m pip install pyarrow
sudo python3 -m pip install boto3
sudo python3 -m pip install s3fs
sudo python3 -m pip install fsspec

sudo python3 -m pip install tensorflow