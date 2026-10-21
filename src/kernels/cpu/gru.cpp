#include "runtime/kernels.h"

#include <cmath>
#include <stdexcept>

namespace runtime::cpu {

namespace {

void validate_gru(
    const Tensor& input,
    const Tensor& hidden,
    const Tensor& input_weights,
    const Tensor& hidden_weights,
    const Tensor& bias) {
  if (input.dtype() != DType::Float32 || hidden.dtype() != DType::Float32 ||
      input_weights.dtype() != DType::Float32 || hidden_weights.dtype() != DType::Float32 ||
      bias.dtype() != DType::Float32 || !input.is_contiguous() || !hidden.is_contiguous() ||
      !input_weights.is_contiguous() || !hidden_weights.is_contiguous() || !bias.is_contiguous()) {
    throw std::invalid_argument("gru_step requires contiguous float32 tensors");
  }
  if (input.rank() != 1 || hidden.rank() != 1 || input_weights.rank() != 2 ||
      hidden_weights.rank() != 2 || bias.rank() != 1) {
    throw std::invalid_argument("gru_step has invalid tensor ranks");
  }
  const std::size_t input_size = input.shape()[0];
  const std::size_t hidden_size = hidden.shape()[0];
  if (input_weights.shape() != Shape{3 * hidden_size, input_size} ||
      hidden_weights.shape() != Shape{3 * hidden_size, hidden_size} ||
      bias.shape() != Shape{3 * hidden_size}) {
    throw std::invalid_argument("gru_step weight shapes do not match input and hidden sizes");
  }
}

}  // namespace

Tensor gru_step(
    const Tensor& input,
    const Tensor& hidden,
    const Tensor& input_weights,
    const Tensor& hidden_weights,
    const Tensor& bias,
    MemoryPool* pool) {
  validate_gru(input, hidden, input_weights, hidden_weights, bias);
  const std::size_t input_size = input.shape()[0];
  const std::size_t hidden_size = hidden.shape()[0];
  Tensor output = pool == nullptr
      ? Tensor::empty({hidden_size}, DType::Float32)
      : pool->allocate_tensor({hidden_size}, DType::Float32);
  const float* input_data = input.data<float>();
  const float* hidden_data = hidden.data<float>();
  const float* input_data_weights = input_weights.data<float>();
  const float* hidden_data_weights = hidden_weights.data<float>();
  const float* bias_data = bias.data<float>();
  float* output_data = output.data<float>();

  for (std::size_t unit = 0; unit < hidden_size; ++unit) {
    float update = bias_data[unit];
    float reset = bias_data[hidden_size + unit];
    float candidate = bias_data[2 * hidden_size + unit];
    for (std::size_t index = 0; index < input_size; ++index) {
      update += input_data_weights[unit * input_size + index] * input_data[index];
      reset += input_data_weights[(hidden_size + unit) * input_size + index] * input_data[index];
      candidate += input_data_weights[(2 * hidden_size + unit) * input_size + index] * input_data[index];
    }
    float recurrent_candidate = 0.0f;
    for (std::size_t index = 0; index < hidden_size; ++index) {
      update += hidden_data_weights[unit * hidden_size + index] * hidden_data[index];
      reset += hidden_data_weights[(hidden_size + unit) * hidden_size + index] * hidden_data[index];
      recurrent_candidate +=
          hidden_data_weights[(2 * hidden_size + unit) * hidden_size + index] * hidden_data[index];
    }
    update = 1.0f / (1.0f + std::exp(-update));
    reset = 1.0f / (1.0f + std::exp(-reset));
    candidate = std::tanh(candidate + reset * recurrent_candidate);
    output_data[unit] = update * hidden_data[unit] + (1.0f - update) * candidate;
  }
  return output;
}

}  // namespace runtime::cpu
