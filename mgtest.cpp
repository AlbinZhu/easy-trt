#include "Inference.h"
#include <opencv2/imgcodecs.hpp>

int main(int argc, char *argv[]) {

  auto img = cv::imread("E:/mgsx/val/1.jpg");

  auto img2 = cv::imread("E:/mgsx/frame2/1.jpg");

  unsigned char *bytes[] = {img.data, img2.data};

  DetResult *res = new DetResult[4];
  init();
  int num = infer(2560, 1440, 3, bytes, 4, res);
}
