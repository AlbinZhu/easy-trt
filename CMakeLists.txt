cmake_minimum_required(VERSION 3.14)

project(easy-trt CXX CUDA)
set(TRT_DIR "D:/TensorRT-8.6.1.6")
include_directories(${TRT_DIR}/include)
include_directories(${TRT_DIR}/samples/common)
# 动态链接库
message(${TRT_DIR}/lib)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
execute_process(
COMMAND sleep 10
)


include_directories(include)
include_directories(third_party/eigen)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(D:/compiled_fastdeploy/third_libs/install/paddle2onnx/include)
link_directories(${TRT_DIR}/lib D:/compiled_fastdeploy/third_libs/install/paddle2onnx/lib)

# link_directories(${TRT_DIR}/lib)
link_directories(D:/TensorRT-8.6.1.6/lib)
# include_directories(fastdeploy/core)
# include_directories(fastdeploy/utils)
# include_directories(fastdeploy/vision/utils)
# include_directories(fastdeploy/runtime)
# include_directories(fastdeploy/vision/common/processors)
# include_directories(fastdeploy/function)
set(OpenCV_DIR "D:/opencv/build/x64/vc16/lib")
# 寻找OpenCV包路径
find_package(OpenCV)
# 标记文件路径
include_directories(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE CC "${PROJECT_SOURCE_DIR}/fastdeploy/*.cc")
file(GLOB_RECURSE CU "${PROJECT_SOURCE_DIR}/fastdeploy/*.cu")

add_executable(${PROJECT_NAME} main.cpp ${CC} ${CU})
# if(MSVC)
# disable warnings for dll export
# target_compile_options(${LIBRARY_NAME} PRIVATE "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:/wd4251>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/wd4251>")
file(GLOB FD_FILES_REQUIRE_BIGOBJ fastdeploy/function/reduce.cc)
set_source_files_properties(${FD_FILES_REQUIRE_BIGOBJ} PROPERTIES COMPILE_FLAGS "/bigobj")
# endif()
# add_executable(${PROJECT_NAME} ${CC})
# add_executable(${PROJECT_NAME} ${CU})

target_link_libraries(${PROJECT_NAME}
        nvinfer
        nvonnxparser
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBS})

