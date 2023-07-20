cmake_minimum_required(VERSION 2.8)
project( tensorCV )

# Make flags
set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g" )
set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -gencode arch=compute_86,code=sm_86 )

# Enable OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Enable CUDA
find_package( CUDA REQUIRED )
include_directories( ${CUDA_INCLUDE_DIRS} )

# Add excutable
cuda_compile(CUDA_OBJECT functions.cu)
set( SOURCES test.cpp )
add_executable( ${PROJECT_NAME} ${SOURCES} ${CUDA_OBJECT})

# Linking
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} -lcublas)