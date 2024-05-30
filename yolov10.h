#pragma once
#include "utils/utils.h"
#include "utils/yolo.h"

class YOLOV8 : public yolo::YOLO {
public:
  YOLOV8(const utils::InitParameter &param);
  ~YOLOV8();
  virtual bool init(const std::vector<unsigned char> &trtFile);
  virtual void preprocess(const std::vector<cv::Mat> &imgsBatch);
  virtual void postprocess(const std::vector<cv::Mat> &imgsBatch);

private:
  float *m_output_src_transpose_device;
};
