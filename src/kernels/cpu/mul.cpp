#include "runtime/kernels.h"

#include <stdexcept>

namespace runtime::cpu {

Tensor mul(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool) {
  if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32 ||
      lhs.shape() != rhs.shape() || !lhs.is_contiguous() || !rhs.is_contiguous()) {
    throw std::invalid_argument("mul requires matching contiguous float32 tensors");
  }
  Tensor output = pool == nullptr ? Tensor::empty(lhs.shape()) : pool->allocate_tensor(lhs.shape());
  for (std::size_t index = 0; index < lhs.numel(); ++index) {
    output.data<float>()[index] = lhs.data<float>()[index] * rhs.data<float>()[index];
  }
  return output;
}

}  // namespace runtime::cpu
