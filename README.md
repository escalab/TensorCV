# TensorCV
This repo conatins artifacts of the TensorCV project.

# Enviroment
```
OS: Ubuntu 20.04 LTS
CPU: Intel Core i9 12900KF
GCC: 9.4.0
GPU: Nvidia RTX 3090
CUDA: 11.7
```

# Build
```
apt update
apt-get install -y wget build-essential git g++ vim htop curl clang cmake make ninja-build unzip 
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DWITH_CUDA=ON ..
cmake --build . -- -j 10 && make install -j 10
```

```
mkdir build && cd build
cmake ..
make
```

# Run
```
./cvTest
```
