#pragma once
#include "../models/det_model.h"
#include "../utils/utils.h"

class YOLOV10 : public model::DetModel {
public:
  YOLOV10(const utils::InitParameter &param) : model::DetModel(param){};
  // virtual bool init(const std::vector<unsigned char> &trtFile);
  // virtual void preprocess(const std::vector<cv::Mat> &imgsBatch);
  virtual void postprocess(const std::vector<cv::Mat> &imgsBatch);

  // private:
  //   float *m_output_src_transpose_device;
};
