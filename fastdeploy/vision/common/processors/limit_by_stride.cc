// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/vision/common/processors/limit_by_stride.h"

namespace fastdeploy {
namespace vision {

bool LimitByStride::ImplByOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetOpenCVMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  int rw = origin_w - origin_w % stride_;
  int rh = origin_h - origin_h % stride_;
  if (rw == 0) {
    rw = stride_;
  }
  if (rh == 0) {
    rh = stride_;
  }
  if (rw != origin_w || rh != origin_w) {
    cv::resize(*im, *im, cv::Size(rw, rh), 0, 0, interp_);
    mat->SetWidth(im->cols);
    mat->SetHeight(im->rows);
  }
  return true;
}

#ifdef ENABLE_FLYCV
bool LimitByStride::ImplByFlyCV(Mat* mat) {
  fcv::Mat* im = mat->GetFlyCVMat();
  int origin_w = im->width();
  int origin_h = im->height();
  int rw = origin_w - origin_w % stride_;
  int rh = origin_h - origin_h % stride_;
  if (rw == 0) {
    rw = stride_;
  }
  if (rh == 0) {
    rh = stride_;
  }
  if (rw != origin_w || rh != origin_h) {
    auto interp_method = fcv::InterpolationType::INTER_LINEAR;
    if (interp_ == 0) {
      interp_method = fcv::InterpolationType::INTER_NEAREST;
    } else if (interp_ == 1) {
      interp_method = fcv::InterpolationType::INTER_LINEAR;
    } else if (interp_ == 2) {
      interp_method = fcv::InterpolationType::INTER_CUBIC;
    } else if (interp_ == 3) {
      interp_method = fcv::InterpolationType::INTER_AREA; 
    } else {
      FDERROR << "LimitByStride: Only support interp_ be 0/1/2/3 with FlyCV, but "
                 "now it's "
              << interp_ << "." << std::endl;
      return false;
    }

    fcv::Mat new_im;
    fcv::resize(*im, new_im, fcv::Size(rw, rh), 0, 0, interp_method);
    mat->SetMat(new_im);
    mat->SetWidth(new_im.width());
    mat->SetHeight(new_im.height());
  }
  return true;
}
#endif

bool LimitByStride::Run(Mat* mat, int stride, int interp, ProcLib lib) {
  auto r = LimitByStride(stride, interp);
  return r(mat, lib);
}
}  // namespace vision
}  // namespace fastdeploy
