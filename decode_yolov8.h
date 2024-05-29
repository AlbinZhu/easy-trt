#pragma once
#include "utils/kernel_function.h"
#include "utils/utils.h"

namespace yolov8 {
void DEPLOY_DECL decodeDevice(utils::InitParameter param, float *src,
                              int srcWidth, int srcHeight, int srcLength,
                              float *dst, int dstWidth, int dstHeight);
void DEPLOY_DECL transposeDevice(utils::InitParameter param, float *src,
                                 int srcWidth, int srcHeight, int srcArea,
                                 float *dst, int dstWidth, int dstHeight);
} // namespace yolov8
