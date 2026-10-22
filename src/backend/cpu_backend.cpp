#include "runtime/cpu_backend.h"

#include <algorithm>
#include <stdexcept>

#include "runtime/kernels.h"

namespace runtime {

std::string_view CPUBackend::name() const {
  return "cpu";
}

Tensor CPUBackend::add(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const {
  // Backend 本身不做具体数值循环，只负责把请求转给对应 kernel。
  return cpu::add(lhs, rhs, &pool);
}

Tensor CPUBackend::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MemoryPool& pool) const {
  return cpu::matmul(lhs, rhs, &pool);
}

Tensor CPUBackend::sigmoid(const Tensor& input, MemoryPool& pool) const {
  return cpu::sigmoid(input, &pool);
}

Tensor CPUBackend::tanh(const Tensor& input, MemoryPool& pool) const {
  return cpu::tanh(input, &pool);
}

Tensor CPUBackend::relu(const Tensor& input, MemoryPool& pool) const {
  if (input.dtype() != DType::Float32 || !input.is_contiguous()) {
    throw std::invalid_argument("relu requires a contiguous float32 tensor");
  }
  Tensor output = pool.allocate_tensor(input.shape(), input.dtype());
  for (std::size_t index = 0; index < input.numel(); ++index) {
    output.data<float>()[index] = std::max(0.0f, input.data<float>()[index]);
  }
  return output;
}

Tensor CPUBackend::conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride,
    std::size_t padding,
    MemoryPool& pool) const {
  return cpu::conv2d(input, weights, bias, stride, padding, &pool);
}

Tensor CPUBackend::conv_transpose2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride,
    std::size_t padding,
    MemoryPool& pool) const {
  return cpu::conv_transpose2d(input, weights, bias, stride, padding, &pool);
}

Tensor CPUBackend::gru_step(
    const Tensor& input,
    const Tensor& hidden,
    const Tensor& input_weights,
    const Tensor& hidden_weights,
    const Tensor& bias,
    MemoryPool& pool) const {
  return cpu::gru_step(input, hidden, input_weights, hidden_weights, bias, &pool);
}

}  // namespace runtime
