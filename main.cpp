#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/runtime/runtime_option.h"
#include "fastdeploy/vision/common/result.h"
#include "fastdeploy/vision/visualize/visualize.h"
#include "fastdeploy/vision/yolov8/yolov8.h"
#include <iostream>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <ostream>
#include <regex>
#include <string>
#include <vector>
int main(int argc, char *argv[]) {
  std::cout << "hello" << std::endl;
  fastdeploy::RuntimeOption options;
  options.UseGpu(0);
  options.UseTrtBackend();
  options.trt_option.SetShape("images", {1, 3, 640, 640}, {1, 3, 640, 640},
                              {4, 3, 640, 640});
  std::string modelPath = "D:/project/ultralytics/train72/weights/best.onnx";
  std::regex toReplace("onnx");
  std::string replaceWidth("cache");
  std::string trt_cache =
      std::regex_replace(modelPath, toReplace, replaceWidth);
  options.trt_option.serialize_file = trt_cache;

  auto model = std::make_unique<fastdeploy::vision::detection::YOLOv8>(
      modelPath, "", options);
  if (!model->Initialized()) {
    std::cerr << "failed to initialize." << std::endl;
    return 0;
  }
  auto img = cv::imread("");
  /* std::vector<fastdeploy::vision::DetectionResult> results; */
  fastdeploy::vision::DetectionResult *result;
  if (!model->Predict(img, result)) {
    std::cerr << "Failed to predict." << std::endl;
  }

  auto visImg = fastdeploy::vision::VisDetection(img, *result);

  return 0;
}
