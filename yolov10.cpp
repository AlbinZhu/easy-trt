#include "yolov10.h"
#include "decode_yolov10.h"

YOLOV10::YOLOV10(const utils::InitParameter &param) : yolo::YOLO(param) {}

bool YOLOV10::init(const std::vector<unsigned char> &trtFile) {
  if (trtFile.empty()) {
    return false;
  }
  std::unique_ptr<nvinfer1::IRuntime> runtime =
      std::unique_ptr<nvinfer1::IRuntime>(
          nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
  if (runtime == nullptr) {
    return false;
  }
  this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(trtFile.data(), trtFile.size()));

  if (this->m_engine == nullptr) {
    return false;
  }
  this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(
      this->m_engine->createExecutionContext());
  if (this->m_context == nullptr) {
    return false;
  }
  if (m_param.dynamic_batch) {
    this->m_context->setInputShape(
        this->m_engine->getIOTensorName(0),
        nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
  }
  m_output_dims =
      this->m_context->getTensorShape(this->m_engine->getIOTensorName(1));
  m_total_objects = m_output_dims.d[1];
  assert(m_param.batch_size <= m_output_dims.d[1]);
  m_output_area = 1;
  for (int i = 1; i < m_output_dims.nbDims; i++) {
    if (m_output_dims.d[i] != 0) {
      m_output_area *= m_output_dims.d[i];
    }
  }
  CHECK(cudaMalloc(&m_output_src_device,
                   m_param.batch_size * m_output_area * sizeof(float)));
  float a = float(m_param.dst_h) / m_param.src_h;
  float b = float(m_param.dst_w) / m_param.src_w;
  float scale = a < b ? a : b;
  cv::Mat src2dst =
      (cv::Mat_<float>(2, 3) << scale, 0.f,
       (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5, 0.f, scale,
       (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
  cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
  std::cout << src2dst.at<float>(0, 1) << "111" << src2dst.at<float>(1, 0)
            << std::endl;
  cv::invertAffineTransform(src2dst, dst2src);

  m_dst2src.v0 = dst2src.ptr<float>(0)[0];
  m_dst2src.v1 = dst2src.ptr<float>(0)[1];
  m_dst2src.v2 = dst2src.ptr<float>(0)[2];
  m_dst2src.v3 = dst2src.ptr<float>(1)[0];
  m_dst2src.v4 = dst2src.ptr<float>(1)[1];
  m_dst2src.v5 = dst2src.ptr<float>(1)[2];

  return true;
}

void YOLOV10::preprocess(const std::vector<cv::Mat> &imgsBatch) {
  resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w,
               m_param.src_h, m_input_resize_device, m_param.dst_w,
               m_param.dst_h, 114, m_dst2src);

  // float *resize_data = new float[3 * 640 * 640];
  // cudaMemcpy(resize_data, m_input_resize_device, sizeof(float) * 3 * 640 *
  // 640,
  //            cudaMemcpyDeviceToHost);
  //
  // cv::Mat resizeImg(cv::Size(640, 640), CV_8UC3, resize_data);
  // cv::imshow("resize", resizeImg);
  // cv::waitKey(0);
  bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w,
                m_param.dst_h, m_input_rgb_device, m_param.dst_w,
                m_param.dst_h);
  normDevice(m_param.batch_size, m_input_rgb_device, m_param.dst_w,
             m_param.dst_h, m_input_norm_device, m_param.dst_w, m_param.dst_h,
             m_param);
  hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.dst_w,
                m_param.dst_h, m_input_hwc_device, m_param.dst_w,
                m_param.dst_h);
}

void YOLOV10::postprocess(const std::vector<cv::Mat> &imgsBatch) {
  yolov10::decodeDevice(m_param, m_output_src_device, 6, m_total_objects,
                        m_output_area, m_output_objects_device,
                        m_output_objects_width, m_param.topK);
  CHECK(cudaMemcpy(m_output_objects_host, m_output_objects_device,
                   m_param.batch_size * sizeof(float) * (1 + 6 * m_param.topK),
                   cudaMemcpyDeviceToHost));
  for (size_t bi = 0; bi < imgsBatch.size(); bi++) {
    int num_boxes =
        std::min((int)(m_output_objects_host +
                       bi * (m_param.topK * m_output_objects_width + 1))[0],
                 m_param.topK);
    for (size_t i = 0; i < num_boxes; i++) {
      float *ptr = m_output_objects_host +
                   bi * (m_param.topK * m_output_objects_width + 1) +
                   m_output_objects_width * i + 1;
      float x_lt = m_dst2src.v0 * ptr[0] + m_dst2src.v1 * ptr[1] + m_dst2src.v2;
      float y_lt = m_dst2src.v3 * ptr[0] + m_dst2src.v4 * ptr[1] + m_dst2src.v5;
      float x_rb = m_dst2src.v0 * ptr[2] + m_dst2src.v1 * ptr[3] + m_dst2src.v2;
      float y_rb = m_dst2src.v3 * ptr[2] + m_dst2src.v4 * ptr[3] + m_dst2src.v5;
      m_objectss[bi].emplace_back(x_lt, y_lt, x_rb, y_rb, ptr[4], (int)ptr[5]);
    }
  }
}
