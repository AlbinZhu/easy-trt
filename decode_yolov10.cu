#include "decode_yolov10.h"

__global__ void decode_yolov10_device_kernel(int batch_size, int num_class,
                                             int topK, float conf_thresh,
                                             float *src, int srcWidth,
                                             int srcHeight, int srcArea,
                                             float *dst, int dstWidth,
                                             int dstHeight, int dstArea) {
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= srcHeight || dy >= batch_size) {
    return;
  }
  float *pitem = src + dy * srcArea + dx * srcWidth;
  float x1 = *pitem++;
  float y1 = *pitem++;
  float x2 = *pitem++;
  float y2 = *pitem++;
  float confidence = *pitem++;
  int label = int(*pitem++);

  if (confidence < conf_thresh) {
    return;
  }

  int index = atomicAdd(dst + dy * dstArea, 1);
  if (index >> topK) {
    return;
  }
  float *pout_item = dst + dy * dstArea + 1 + index * dstWidth;
  *pout_item++ = x1;
  *pout_item++ = y1;
  *pout_item++ = x2;
  *pout_item++ = y2;
  *pout_item++ = confidence;
  *pout_item++ = label;
  //*pout_item++ = 1;
}

void yolov10::decodeDevice(utils::InitParameter param, float *src, int srcWidth,
                           int srcHeight, int srcArea, float *dst, int dstWidth,
                           int dstHeight) {
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size((srcHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int dstArea = 1 + dstWidth * dstHeight;

  decode_yolov10_device_kernel<<<grid_size, block_size, 0, nullptr>>>(
      param.batch_size, param.num_class, param.topK, param.conf_thresh, src,
      srcWidth, srcHeight, srcArea, dst, dstWidth, dstHeight, dstArea);
}

__global__ void transpose_device_kernel(int batch_size, float *src,
                                        int srcWidth, int srcHeight,
                                        int srcArea, float *dst, int dstWidth,
                                        int dstHeight, int dstArea) {
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= dstHeight || dy >= batch_size) {
    return;
  }
  float *p_dst_row = dst + dy * dstArea + dx * dstWidth;
  float *p_src_col = src + dy * srcArea + dx;

  for (int i = 0; i < dstWidth; i++) {
    p_dst_row[i] = p_src_col[i * srcWidth];
  }
}

void yolov10::transposeDevice(utils::InitParameter param, float *src,
                              int srcWidth, int srcHeight, int srcArea,
                              float *dst, int dstWidth, int dstHeight) {
  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size((dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int dstArea = dstWidth * dstHeight;

  transpose_device_kernel<<<grid_size, block_size, 0, nullptr>>>(
      param.batch_size, src, srcWidth, srcHeight, srcArea, dst, dstWidth,
      dstHeight, dstArea);
}
