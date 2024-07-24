#include "yolov10.h"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
typedef struct {
  int idx;
  int cls;
  float score;
  int box[4] = {0, 0, 0, 0};
} DetResult;

class InferEngine {
public:
  InferEngine() { init(); }
  int run(std::vector<cv::Mat> imgs, DetResult *results);

  int binThres;
  int areaThres;

private:
  int debug;

  std::string savePath;
  std::string modelPath;
  std::string bsModelPath;

  double confThres;
  double bsThres;
  // double iouThres;

  int maxResults;

  int w;
  int h;

  std::shared_ptr<YOLOV10> model;
  std::shared_ptr<YOLOV10> bsModel;

  void init();
  void loadConfig();
};

extern "C" __declspec(dllexport) void init();
extern "C" __declspec(dllexport) int infer(int width, int height, int channel,
                                           unsigned char *bytes[], int count,
                                           DetResult *results);

extern "C" __declspec(dllexport) int compare(int width, int height, int channel,
                                             unsigned char *bytes[], int x,
                                             int y, int w, int h);
int inferTest(std::vector<cv::Mat> imgs, DetResult *results);
