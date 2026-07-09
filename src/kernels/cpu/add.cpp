#include "runtime/kernels.h"

#include <stdexcept>

namespace runtime::cpu {

Tensor add(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool) {
  if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32) {
    throw std::runtime_error("cpu::add only supports float32");
  }
  if (lhs.shape() != rhs.shape()) {
    throw std::runtime_error("cpu::add requires matching shapes");
  }
  if (!lhs.is_contiguous() || !rhs.is_contiguous()) {
    throw std::runtime_error("cpu::add currently requires contiguous tensors");
  }

  Tensor out = pool ? pool->allocate_tensor(lhs.shape(), lhs.dtype())
                    : Tensor::empty(lhs.shape(), lhs.dtype());

  const float* lhs_data = lhs.data<float>();
  const float* rhs_data = rhs.data<float>();
  float* out_data = out.data<float>();

  for (std::size_t i = 0; i < lhs.numel(); ++i) {
    out_data[i] = lhs_data[i] + rhs_data[i];
  }

  return out;
}

}  // namespace runtime::cpu
