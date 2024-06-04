#pragma once
#include "base_model.h"

namespace model {
class DetModel : public BaseModel {
public:
  DetModel(const utils::InitParameter &param);
  ~DetModel();

public:
  // virtual bool init(const std::vector<unsigned char> &trtFile);
  // virtual void check();
  // virtual void copy(const std::vector<cv::Mat> &imgsBatch);
  // virtual void preprocess(const std::vector<cv::Mat> &imgsBatch);
  // virtual bool infer();
  virtual void postprocess(const std::vector<cv::Mat> &imgsBatch);
  virtual void reset();

public:
  std::vector<std::vector<utils::Box>> getObjectss() const;

protected:
  std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
  utils::InitParameter m_param;
  // nvinfer1::Dims m_output_dims;
  // int m_output_area;
  // int m_total_objects;
  std::vector<std::vector<utils::Box>> m_objectss;
  // utils::AffineMat m_dst2src;

  // input
  // unsigned char *m_input_src_device;
  // float *m_input_resize_device;
  // float *m_input_rgb_device;
  // float *m_input_norm_device;
  // float *m_input_hwc_device;
  // output
  // float *m_output_src_device;
  // float *m_output_objects_device;
  // float *m_output_objects_host;
  // int m_output_objects_width;
  int *m_output_idx_device;
  float *m_output_conf_device;
};
} // namespace model
