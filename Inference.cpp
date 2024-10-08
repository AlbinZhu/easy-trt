#include "Inference.h"
#include "SimpleIni.h"
#include "logger.h"
#include "utils/utils.h"
#include "yolov10.h"
#include <iostream>
#include <memory>
#include <minwindef.h>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

std::shared_ptr<InferEngine> inferEngine;

void InferEngine::loadConfig() {
  std::cout << "load config" << std::endl;
  CSimpleIni config;
  SI_Error rc;
  rc = config.LoadFile("config.ini");
  if (rc != SI_OK) {
    std::cerr << "Unable to load config file" << std::endl;
    throw "Unable to load config file";
  }

  debug = config.GetLongValue("config", "debug");

  modelPath = config.GetValue("config", "model_path");
  bsModelPath = config.GetValue("config", "bs_model_path");
  confThres = config.GetDoubleValue("config", "conf_thres");
  bsThres = config.GetDoubleValue("config", "bs_conf_thres");
  std::cout << "model path " << modelPath << std::endl;
  // iouThres = config.GetDoubleValue("config", "iou_thres");
  maxResults = config.GetLongValue("config", "max_results");

  binThres = config.GetLongValue("config", "bin_thres");
  areaThres = config.GetLongValue("config", "area_thres");

  w = config.GetLongValue("config", "w");
  h = config.GetLongValue("config", "h");
  std::cout << "max  " << maxResults << std::endl;
}

void InferEngine::init() {
  loadConfig();
  utils::InitParameter param;
  param.num_class = 1;
  param.class_names = utils::dataSets::mg;
  param.input_output_names = {"images", "output0"};
  param.src_h = h;
  param.src_w = w;
  param.dst_h = 640;
  param.dst_w = 640;
  param.batch_size = 1;
  param.iou_thresh = confThres;
  param.conf_thresh = 0.6;
  param.is_show = false;
  param.is_save = false;

  // model = YOLOV10(param);
  model = std::make_shared<YOLOV10>(param);
  std::vector<unsigned char> trt_file = utils::loadModel(modelPath);
  if (trt_file.empty()) {
    sample::gLogError << "trt_file is empty!" << std::endl;
    // return -1;
  }
  if (!model->init(trt_file)) {
    sample::gLogError << "initEngine() ocur errors!" << std::endl;
    // return -1;
  }
  model->check();
  param.conf_thresh = bsThres;
  bsModel = std::make_unique<YOLOV10>(param);
  std::vector<unsigned char> bs_trt_file = utils::loadModel(bsModelPath);
  if (bs_trt_file.empty()) {
    sample::gLogError << "bs trt_file is empyt!" << std::endl;
  }

  if (!bsModel->init(bs_trt_file)) {
    sample::gLogError << "bs initEngine() ocur errors!" << std::endl;
  }
  bsModel->check();
}

int InferEngine::run(std::vector<cv::Mat> imgs, DetResult *results) {
  // utils::DeviceTimer d_t0;

  std::thread bs([this, imgs] {
    bsModel->copy(imgs);
    bsModel->preprocess(imgs);
    bsModel->infer();
    bsModel->postprocess(imgs);
  });

  model->copy(imgs);
  model->preprocess(imgs);
  model->infer();
  model->postprocess(imgs);
  // utils::show(model->getObjectss(), utils::dataSets::mg, 0, imgs);
  // utils::save(model->getObjectss(), utils::dataSets::mg, savePath, imgs, 2,
  // 1);

  int total = 0;
  for (int i = 0; i < model->getObjectss().size(); i++) {
    auto boxes = model->getObjectss()[i];
    for (int j = 0; j < boxes.size(); j++) {
      if (total >= maxResults) {
        break;
      }
      auto box = boxes[j];
      results[total].idx = i;
      results[total].cls = 0;
      results[total].score = box.confidence;
      results[total].box[0] = box.left;
      results[total].box[1] = box.top;
      results[total].box[2] = box.right;
      results[total].box[3] = box.bottom;

      total += 1;
    }
  }
  bs.join();
  // utils::show(bsModel->getObjectss(), utils::dataSets::mg, 0, imgs);
  // utils::save(bsModel->getObjectss(), utils::dataSets::mg, "000.jpg", imgs,
  // 1, 0);
  for (int i = 0; i < bsModel->getObjectss().size(); i++) {
    auto boxes = bsModel->getObjectss()[i];
    for (int j = 0; j < boxes.size(); j++) {
      if (total >= maxResults) {
        break;
      }
      auto box = boxes[j];
      results[total].idx = i;
      results[total].cls = 1;
      results[total].score = box.confidence;
      results[total].box[0] = box.left;
      results[total].box[1] = box.top;
      results[total].box[2] = box.right;
      results[total].box[3] = box.bottom;

      total += 1;
    }
  }
  model->resetModel();
  bsModel->resetModel();
  return total;
}

void init() { inferEngine = std::make_shared<InferEngine>(); }

int infer(int width, int height, int channel, unsigned char *bytes[], int count,
          DetResult *results) {
  std::cout << "infer" << std::endl;
  std::vector<cv::Mat> imgs;
  for (int i = 0; i < count; i++) {
    cv::Mat img(cv::Size(width, height), CV_8UC3, bytes[i]);
    // cv::imshow("img", img);
    // cv::waitKey(0);
    imgs.emplace_back(img);
  }
  int res = inferEngine->run(imgs, results);
  return res;
}

int compare(int width, int height, int channel, unsigned char *bytes[], int x,
            int y, int w, int h) {
  cv::Mat img1(cv::Size(width, height), CV_8UC3, bytes[0]);
  cv::Mat img2(cv::Size(width, height), CV_8UC3, bytes[1]);

  auto sub1 = img1(cv::Rect(x, y, w, h));
  auto sub2 = img2(cv::Rect(x, y, w, h));

  cv::cvtColor(sub1, sub1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(sub2, sub2, cv::COLOR_BGR2GRAY);

  cv::blur(sub1, sub1, cv::Size(3, 3));
  cv::blur(sub2, sub2, cv::Size(3, 3));

  cv::Mat resImg;
  cv::absdiff(sub1, sub2, resImg);

  cv::threshold(resImg, resImg, inferEngine->binThres, 255, cv::THRESH_BINARY);

  int num = cv::countNonZero(resImg);
  if (num > inferEngine->areaThres) {
    return 1;
  } else {
    return 0;
  }
  // return 0;
}

// int inferTest(std::vector<cv::Mat> imgs, DetResult *results) {}
