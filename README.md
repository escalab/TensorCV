# TensorCV
This repo contains artifacts of the TensorCV project.

# Environment
```
OS: Ubuntu 20.04 LTS
CPU: Intel Core i9 12900KF
GPU: Nvidia RTX 3090
GCC: 9.4.0
CUDA: 11.7
```

# Build
Setup environment
We recommend you to use the docker image nvidia/cuda:11.7.0-devel-ubuntu20.04 
(https://hub.docker.com/r/nvidia/cuda)
```
apt update
apt-get install -y build-essential git cmake make
```

Install CUDA (https://developer.nvidia.com/cuda-downloads)

Install opencv and opencv_contrib with cuda support (https://opencv.org/)
```
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv && mkdir build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DWITH_CUDA=ON ..
cmake --build . -- -j && make install -j
```

Build TensorCV
```
git clone https://github.com/escalab/TensorCV.git
cd TensorCV && mkdir build && cd build
cmake ..
make
```

# Run
```
./cvTest CV 4032 # Run TensorCV part only with 4032x3024 input images
./cvTest <MODE> <SIZE>
MODE: CPU/GPU/CV/ALL
SIZE: 480 / 1600 / 2048 / 2592 / 3264 / 4032
```

This repository only provides 20 images as samples
|SIZE | Input image size (directory) |
|-|-|
|`480`  | 480 x 320 (img/480)
|`1600` | 1600 x 1200 (img/1600)
|`2048` | 2048 x 1536 (img/2048)
|`2592` | 2592 x 1936 (img/2592)
|`3264` | 3264 x 2448 (img/3264)
|`4032` | 4032 x 3024 (img/4032)
