#include "Inference.h"
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char *argv[]) {

  auto img = cv::imread("E:/mgsx/val/96.jpg");

  auto img2 = cv::imread("E:/mgsx/val/97.jpg");

  // unsigned char *bytes[] = {img.data, img2.data};

  // DetResult *res = new DetResult[4];
  // init();
  // int num = infer(2560, 1440, 3, bytes, 4, res);

  // auto sub1 = img(cv::Rect(968, 661, 1051 - 968, 798 - 661));
  // auto sub2 = img2(cv::Rect(968, 661, 1051 - 968, 798 - 661));
  // cv::imwrite("sub1.jpg", sub1);
  // cv::imwrite("sub2.jpg", sub2);
  // cv::imshow("img1", sub1);
  // cv::imshow("img2", sub2);
  // cv::waitKey(0);
  init();
  // unsigned char *bytes2[] = {sub1.data, sub2.data};
  int size = img.total() * img.elemSize();
  typedef unsigned char byte;
  byte *bytes = new byte[size];
  std::memcpy(bytes, img.data, size * sizeof(byte));
  // cv::Mat test(cv::Size(2560, 1440), CV_8UC3, bytes);
  cv::Mat timg = cv::Mat(cv::Size(2560, 1440), CV_8UC3, bytes);

  byte *bytes2 = new byte[size];
  std::memcpy(bytes2, img2.data, size * sizeof(byte));

  cv::imshow("test", timg);
  cv::waitKey(0);

  byte *ba[] = {bytes, bytes2};
  // int resn = compare(2560, 1440, 3, bytes2, 968, 661, 1051 - 968, 798 - 661);
  int resn = compare(2560, 1440, 3, ba, 968, 661, 1051 - 968, 798 - 661);

  std::cout << "res  " << resn << std::endl;
}