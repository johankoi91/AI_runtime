#include "runtime/kernels.h"

#include <cmath>
#include <stdexcept>

namespace runtime::cpu {

namespace {

template <typename Function>
Tensor apply(const Tensor& input, MemoryPool* pool, Function function) {
  if (input.dtype() != DType::Float32 || !input.is_contiguous()) {
    throw std::runtime_error("activation requires a contiguous float32 tensor");
  }
  Tensor output = pool == nullptr
      ? Tensor::empty(input.shape(), input.dtype())
      : pool->allocate_tensor(input.shape(), input.dtype());
  const float* input_data = input.data<float>();
  float* output_data = output.data<float>();
  for (std::size_t index = 0; index < input.numel(); ++index) {
    output_data[index] = function(input_data[index]);
  }
  return output;
}

}  // namespace

Tensor sigmoid(const Tensor& input, MemoryPool* pool) {
  return apply(input, pool, [](float value) { return 1.0f / (1.0f + std::exp(-value)); });
}

Tensor tanh(const Tensor& input, MemoryPool* pool) {
  return apply(input, pool, [](float value) { return std::tanh(value); });
}

}  // namespace runtime::cpu
