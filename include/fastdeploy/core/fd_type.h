
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

#include <ostream>
#include <sstream>
#include <string>

#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

enum FDDataType {
  BOOL,
  INT16,
  INT32,
  INT64,
  FP16,
  FP32,
  FP64,
  UNKNOWN1,
  UNKNOWN2,
  UNKNOWN3,
  UNKNOWN4,
  UNKNOWN5,
  UNKNOWN6,
  UNKNOWN7,
  UNKNOWN8,
  UNKNOWN9,
  UNKNOWN10,
  UNKNOWN11,
  UNKNOWN12,
  UNKNOWN13,
  UINT8,
  INT8
};

std::ostream &operator<<(std::ostream &out, const FDDataType &fdt);

std::string Str(const FDDataType &fdt);

int32_t FDDataTypeSize(const FDDataType &data_dtype);

template <typename PlainType> struct TypeToDataType {
  static const FDDataType dtype;
};

} // namespace fastdeploy
