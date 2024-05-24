cmake_minimum_required(VERSION 3.12.0)
project(easy-trt VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit)

# set(CMAKE_CUDA_ARCHITECTURES "native")

include(CheckLanguage)
check_language(CUDA)

set(OpenCV_DIR "D:/opencv/build")
# 设置为TensorRT的根目录
set(TRT_DIR "D:/TensorRT-8.6.1.6")

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# find_package(CUDA REQUIRED)

# TensorRT
# 标志文件
include_directories(${TRT_DIR}/include)
include_directories(${TRT_DIR}/samples/common)
# 动态链接库
link_directories(${TRT_DIR}/lib)

# link_directories(${TENSORRT_ROOT}/lib)
# OpencV
# 寻找OpenCV包路径
find_package(OpenCV)
# 标记文件路径
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE CPPS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
  )




add_library(${PROJECT_NAME} ${CPPS})
# add_executable(${PROJECT_NAME} ${CPPS})
target_link_libraries(${PROJECT_NAME}
        nvinfer
        nvonnxparser
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS})
# add_executable(u2net_test test.cu)
# add_executable(${PROJECT_NAME} u2net.cu)
# target_link_libraries(u2net_test
#         nvinfer
#         nvonnxparser
#         # ${CUDA_LIBRARIES}
#         ${OpenCV_LIBS}
#         CUDA::cudart
#         D:/project/duizhong/build/u2net.lib)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
# target_link_libraries(u2net_test )
# target_compile_options(u2net_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# endif()
