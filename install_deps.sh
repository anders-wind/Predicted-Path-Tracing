#!/bin/bash
# if you want to run in docker you should use nvidia-docker 

# Install Base dependencies
sudo apt-get update
sudp apt-get install -y git wget curl
sudo apt-get install -y gcc-8 g++-8 build-essential cmake
sudo apt-get install -y python3.7 pip3 ipython

# Install OpenGL
sudo apt-get install -y xorg openbox doxygen libx11-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev
sudo apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev

# Install CUDA
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run
chmod +x local_installers/cuda_10.1.168_418.67_linux.run
./local_installers/cuda_10.1.168_418.67_linux.run
rm local_installers/cuda_10.1.168_418.67_linux.run

# Install anaconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda-latest-Linux-x86_64.sh
./Miniconda-latest-Linux-x86_64.sh
rm Miniconda-latest-Linux-x86_64.sh

# Get submodules and init them.
git submodule init && git submodule update