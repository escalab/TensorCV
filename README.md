# TensorCV
This repo conatins artifacts of the TensorCV project.

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
```
apt update
apt-get install -y build-essential git cmake make
```

Install CUDA (https://developer.nvidia.com/cuda-downloads)

Install opencv opencv_contrib with cuda support (https://opencv.org/)
```
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv && mkdir build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DWITH_CUDA=ON ..
cmake --build . -- -j && make install -j
```

Install TensorCV
```
git clone https://github.com/escalab/TensorCV.git
cd TensorCV && mkdir build && cd build
cmake ..
make
```

# Run
```
./cvTest
```
