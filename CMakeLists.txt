cmake_minimum_required(VERSION 3.14)

project(easy-trt CXX)
set(TRT_DIR "/home/swing/Documents/TensorRT-10.0.1.6")
include_directories(${TRT_DIR}/include)
include_directories(${TRT_DIR}/samples/common)
# 动态链接库
link_directories(${TRT_DIR}/lib)

set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)


include_directories(fastdeploy/core)
include_directories(fastdeploy/utils)
include_directories(fastdeploy/vision/utils)
include_directories(fastdeploy/runtime)
include_directories(fastdeploy/vision/common/processors)
include_directories(fastdeploy/function)

# 寻找OpenCV包路径
find_package(OpenCV)
# 标记文件路径
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        nvinfer
        nvonnxparser
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS})

