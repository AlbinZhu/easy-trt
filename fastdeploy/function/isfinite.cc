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

#include "fastdeploy/function/isfinite.h"
#include "fastdeploy/core/float16.h"
#include <algorithm>
#include <type_traits>

namespace fastdeploy {
namespace function {

template <typename T, typename OutT, class Enable = void> struct IsNanFunctor {
  OutT operator()(const T& a) const { return static_cast<OutT>(std::isnan(a)); }
};

template <typename T, typename OutT>
struct IsNanFunctor<T, OutT,
                    typename std::enable_if<std::is_integral<T>::value>::type> {
  OutT operator()(const T& a) const { return static_cast<OutT>(false); }
};

template <typename OutT> struct IsNanFunctor<fastdeploy::float16, OutT, void> {
  OutT operator()(const fastdeploy::float16& a) const {
    return static_cast<OutT>(fastdeploy::isnan(a));
  }
};

template <typename T, typename OutT, class Enable = void> struct IsInfFunctor {
  OutT operator()(const T& a) const { return static_cast<OutT>(std::isinf(a)); }
};

template <typename T, typename OutT>
struct IsInfFunctor<T, OutT,
                    typename std::enable_if<std::is_integral<T>::value>::type> {
  OutT operator()(const T& a) const { return static_cast<OutT>(false); }
};

template <typename OutT> struct IsInfFunctor<fastdeploy::float16, OutT, void> {
  OutT operator()(const fastdeploy::float16& a) const {
    return static_cast<OutT>(fastdeploy::isinf(a));
  }
};

template <typename T, typename OutT, class Enable = void>
struct IsFiniteFunctor {
  OutT operator()(const T& a) const {
    return static_cast<OutT>(std::isfinite(a));
  }
};

template <typename T, typename OutT>
struct IsFiniteFunctor<
    T, OutT, typename std::enable_if<std::is_integral<T>::value>::type> {
  OutT operator()(const T& a) const { return static_cast<OutT>(true); }
};

template <typename OutT>
struct IsFiniteFunctor<fastdeploy::float16, OutT, void> {
  OutT operator()(const fastdeploy::float16& a) const {
    return static_cast<OutT>(fastdeploy::isfinite(a));
  }
};

#define DEFINE_ISFINITE_KERNEL(isfinite_kernel, functor)                       \
  template <typename T>                                                        \
  void isfinite_kernel(const FDTensor& x, FDTensor* out, FDDataType dtype) {   \
    FD_VISIT_ALL_TYPES(dtype, #isfinite_kernel, ([&] {                         \
                         out->Allocate(x.Shape(), dtype);                      \
                         functor<T, data_t> unary_func;                        \
                         data_t* out_ptr =                                     \
                             reinterpret_cast<data_t*>(out->Data());           \
                         const T* input_ptr =                                  \
                             reinterpret_cast<const T*>(x.Data());             \
                         std::transform(input_ptr, input_ptr + x.Numel(),      \
                                        out_ptr, unary_func);                  \
                       }));                                                    \
  }

DEFINE_ISFINITE_KERNEL(IsNanKernel, IsNanFunctor)
DEFINE_ISFINITE_KERNEL(IsInfKernel, IsInfFunctor)
DEFINE_ISFINITE_KERNEL(IsFiniteKernel, IsFiniteFunctor)
#undef DEFINE_ISFINITE_KERNEL

void IsNan(const FDTensor& x, FDTensor* out, FDDataType dtype) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "IsNanKernel",
                       ([&] { IsNanKernel<data_t>(x, out, dtype); }));
}

void IsInf(const FDTensor& x, FDTensor* out, FDDataType dtype) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "IsInfKernel",
                       ([&] { IsInfKernel<data_t>(x, out, dtype); }));
}

void IsFinite(const FDTensor& x, FDTensor* out, FDDataType dtype) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "IsFiniteKernel",
                       ([&] { IsFiniteKernel<data_t>(x, out, dtype); }));
}

}  // namespace function
}  // namespace fastdeploy