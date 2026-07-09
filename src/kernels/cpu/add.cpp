#include "runtime/kernels.h"

#include <stdexcept>

namespace runtime::cpu {

Tensor add(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool) {
  // 当前版本只支持 float32，是为了先把“执行路径”讲清楚。
  if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32) {
    throw std::runtime_error("cpu::add only supports float32");
  }

  // 逐元素加法要求形状完全一样。
  if (lhs.shape() != rhs.shape()) {
    throw std::runtime_error("cpu::add requires matching shapes");
  }

  // 为了简化实现，这里只处理连续内存。
  // 这样后面的循环可以直接按一维数组线性访问。
  if (!lhs.is_contiguous() || !rhs.is_contiguous()) {
    throw std::runtime_error("cpu::add currently requires contiguous tensors");
  }

  // 如果给了 MemoryPool，就复用池子；否则就单独新开一块 tensor。
  Tensor out = pool ? pool->allocate_tensor(lhs.shape(), lhs.dtype())
                    : Tensor::empty(lhs.shape(), lhs.dtype());

  const float* lhs_data = lhs.data<float>();
  const float* rhs_data = rhs.data<float>();
  float* out_data = out.data<float>();

  // 逐元素相加是最直观的 kernel 例子。
  for (std::size_t i = 0; i < lhs.numel(); ++i) {
    out_data[i] = lhs_data[i] + rhs_data[i];
  }

  return out;
}

}  // namespace runtime::cpu
