cmake_minimum_required(VERSION 3.12.0)
project(easy-trt VERSION 0.1.0 LANGUAGES C CXX CUDA)

# find_package(CUDA REQUIRED)

# set(CMAKE_CUDA_ARCHITECTURES "native")

include(CheckLanguage)
check_language(CUDA)
set(OpenCV_DIR "D:/opencv/build")
# 设置为TensorRT的根目录

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
find_package(CUDAToolkit)

list(APPEND CUDA_LIBS
  ${CUDA_LIBRARIES}
  ${CUDA_cublas_LIBRARY}
  ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} ${CUDA_nppidei_LIBRARY} ${CUDA_nppial_LIBRARY})

# find_package(CUDA REQUIRED)

# TensorRT
# 标志文件
set(TRT_DIR "D:/TensorRT-8.6.1.6")
include_directories(${TRT_DIR}/include)
include_directories(${TRT_DIR}/samples/common)
# 动态链接库
link_directories(${TRT_DIR}/lib)
message(STATUS CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR})
# link_directories(${TENSORRT_ROOT}/lib)
# OpencV
# 寻找OpenCV包路径
find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})

# include_directories(${CMAKE_CURRENT_SOURCE_DIR})


# file(GLOB CPPS
#   ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cu
#   ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
# )

message(${CMAKE_CURRENT_SOURCE_DIR})

# list(REMOVE_ITEM ${CPPS} main.cpp)
message(${OpenCV_LIBS})
add_library(${PROJECT_NAME} SHARED
        # ${CPPS}
        utils/kernel_function.cu
        # utils/utils.cpp
        # utils/yolo.cpp
        decode_yolov10.cu
        ${TRT_DIR}/samples/common/logger.cpp
        # yolov8.cpp
        # ${TRT_DIR}/samples/common/sampleOptions.cpp
)
# cuda_add_executable(${PROJECT_NAME} ${CPPS})
target_link_libraries(${PROJECT_NAME}
        nvinfer
        nvonnxparser
        ${CUDA_LIBS}
        # ${CUDA_LIBRARIES}
        ${OpenCV_LIBS})

file(GLOB CPPS
${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp})

add_executable(app_yolov10 main.cpp
${CPPS}
models/base_model.cpp
models/det_model.cpp
models/yolov10.cpp
# utils/yolo.cpp
utils/utils.cpp
# yolov10.cpp
        ${TRT_DIR}/samples/common/logger.cpp
)
target_link_libraries(app_yolov10
        D:/project/easy-trt/build/easy-trt.lib
        nvinfer
        nvonnxparser
        ${CUDA_LIBS}
        # ${CUDA_LIBRARIES}CUDA::cudart
        #
        CUDA::cudart
        ${OpenCV_LIBS})
