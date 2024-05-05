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

#pragma once

#include "cast.h"
#include "center_crop.h"
#include "color_space_convert.h"
#include "convert.h"
#include "convert_and_permute.h"
#include "crop.h"
#include "hwc2chw.h"
#include "limit_by_stride.h"
#include "limit_short.h"
#include "normalize.h"
#include "normalize_and_permute.h"
#include "pad.h"
#include "pad_to_size.h"
#include "resize.h"
#include "resize_by_short.h"
#include "stride_pad.h"
#include "warp_affine.h"
#include <unordered_set>

namespace fastdeploy {
namespace vision {

void FuseTransforms(std::vector<std::shared_ptr<Processor>> *processors);
// Fuse Normalize + Cast(Float) to Normalize
void FuseNormalizeCast(std::vector<std::shared_ptr<Processor>> *processors);
// Fuse Normalize + HWC2CHW to NormalizeAndPermute
void FuseNormalizeHWC2CHW(std::vector<std::shared_ptr<Processor>> *processors);
// Fuse Normalize + Color Convert
void FuseNormalizeColorConvert(
    std::vector<std::shared_ptr<Processor>> *processors);

} // namespace vision
} // namespace fastdeploy
