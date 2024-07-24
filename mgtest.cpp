#include "Inference.h"
#include <ctime>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char *argv[]) {

  auto img = cv::imread("E:/202/biansi/val/1-151.jpg");

  auto img2 = cv::imread("E:/202/biansi/val/1-151.jpg");

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
  // cv::Mat test(cv::Size(2560, 1440), CV_8UC3, bytes);
  // cv::Mat timg = cv::Mat(cv::Size(2560, 1440), CV_8UC3, bytes);

  // byte *bytes2 = new byte[size];
  // std::memcpy(bytes2, img2.data, size * sizeof(byte));

  // cv::imshow("test", timg);
  // cv::waitKey(0);

  // int resn = compare(2560, 1440, 3, bytes2, 968, 661, 1051 - 968, 798 - 661);
  int size = img.total() * img.elemSize();
  typedef unsigned char byte;
  byte *bytes = new byte[size];
  std::memcpy(bytes, img.data, size * sizeof(byte));
  byte *ba[] = {bytes};

  clock_t ts = clock();
  for (int i = 0; i < 100; i++) {

    clock_t start = clock();
    DetResult *detresults = new DetResult[10];
    auto num = infer(1920, 1080, 3, ba, 1, detresults);
    clock_t end = clock();
    std::cout << "duration: " << end - start << std::endl;
    // for (int i = 0; i < num; i++) {
    //   auto res = detresults[i];
    //   std::cout << "cls: " << res.cls << std::endl;
    //   std::cout << "score: " << res.score << std::endl;
    //   std::cout << "idx: " << res.idx << std::endl;
    //   std::cout << "x1: " << res.box[0] << std::endl;
    //   std::cout << "y1: " << res.box[1] << std::endl;
    //   std::cout << "x2: " << res.box[2] << std::endl;
    //   std::cout << "y2: " << res.box[3] << std::endl;
    // }
    //
    // std::cout << "res num " << num << std::endl;
  }
  clock_t te = clock();
  std::cout << "total duration: " << te - ts << std::endl;
  // int resn = compare(2560, 1440, 3, ba, 968, 661, 1051 - 968, 798 - 661);

  // std::cout << "res  " << resn << std::endl;
}
