#pragma once
#include "../utils/common_include.h"
#include "../utils/kernel_function.h"
#include "../utils/utils.h"
#include <string>
#include <vector>

namespace model {
class BaseModel {
public:
  // BaseModel();
  // ~BaseModel();

public:
  virtual bool init(const std::vector<unsigned char> &trtFile);
  virtual void check();
  virtual void copy(const std::vector<cv::Mat> &imgsBatch);
  virtual void preprocess(const std::vector<cv::Mat> &imgsBatch);
  virtual bool infer();
  // virtual void postprocess(const std::vector<cv::Mat> &imgsBatch);
  // virtual void reset();

public:
  std::vector<std::vector<utils::Box>> getObjectss() const;

protected:
  std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
  utils::InitParameter m_param;
  int src_w;
  int src_h;
  int dst_w;
  int dst_h;
  int batch_size;
  nvinfer1::Dims m_output_dims;

  bool dynamic = false;
  float scale = 255.f;
  std::vector<float> means = {0.f, 0.f, 0.f};
  std::vector<float> stds = {1.f, 1.f, 1.f};

  int m_output_area;
  int m_total_objects;
  // std::vector<std::vector<utils::Box>> m_objectss;
  utils::AffineMat m_dst2src;

  std::vector<std::string> input_output_names;
  // input
  unsigned char *m_input_src_device;
  float *m_input_resize_device;
  float *m_input_rgb_device;
  float *m_input_norm_device;
  float *m_input_hwc_device;
  // output
  float *m_output_src_device;
  float *m_output_objects_device;
  float *m_output_objects_host;
  int m_output_objects_width;
  // int *m_output_idx_device;
  // float *m_output_conf_device;
};
} // namespace model
