#include "runtime/cpu_backend.h"

#include <algorithm>
#include <stdexcept>

#include "runtime/kernels.h"
#include "runtime/spectral.h"

namespace runtime {

std::string_view CPUBackend::name() const {
  return "cpu";
}

Tensor CPUBackend::add(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const {
  // Backend 本身不做具体数值循环，只负责把请求转给对应 kernel。
  return cpu::add(lhs, rhs, &pool);
}

Tensor CPUBackend::mul(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const {
  return cpu::mul(lhs, rhs, &pool);
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
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t padding_h,
    std::size_t padding_w,
    std::size_t groups,
    MemoryPool& pool) const {
  return cpu::conv2d(input, weights, bias, stride_h, stride_w, padding_h, padding_w, groups, &pool);
}

Tensor CPUBackend::conv_transpose2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t padding_h,
    std::size_t padding_w,
    std::size_t groups,
    MemoryPool& pool) const {
  return cpu::conv_transpose2d(input, weights, bias, stride_h, stride_w, padding_h, padding_w, groups, &pool);
}

Tensor CPUBackend::stft(
    const Tensor& samples,
    std::size_t fft_size,
    std::size_t hop_size,
    MemoryPool& pool) const {
  if (samples.rank() != 1 || samples.dtype() != DType::Float32 || !samples.is_contiguous()) {
    throw std::invalid_argument("cpu::stft requires a contiguous float32 vector");
  }
  std::vector<float> values(samples.data<float>(), samples.data<float>() + samples.numel());
  const Spectrogram result = runtime::stft(values, fft_size, hop_size);
  std::vector<float> output;
  output.reserve(result.values.size() * 2);
  for (const Complex32& value : result.values) {
    output.push_back(value.real);
    output.push_back(value.imag);
  }
  Tensor tensor = pool.allocate_tensor({result.values.size() / result.bins, result.bins, 2});
  std::copy(output.begin(), output.end(), tensor.data<float>());
  return tensor;
}

Tensor CPUBackend::erb_filterbank(
    const Tensor& spectrum,
    const Tensor& filterbank,
    MemoryPool& pool) const {
  return cpu::erb_filterbank(spectrum, filterbank, &pool);
}

Tensor CPUBackend::istft(
    const Tensor& spectrum,
    std::size_t fft_size,
    std::size_t hop_size,
    std::size_t original_samples,
    MemoryPool& pool) const {
  return cpu::istft(spectrum, fft_size, hop_size, original_samples, &pool);
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

Tensor CPUBackend::gru_sequence(
    const Tensor& sequence,
    const Tensor& initial_hidden,
    const Tensor& input_weights,
    const Tensor& hidden_weights,
    const Tensor& bias,
    MemoryPool& pool) const {
  if (sequence.rank() != 3 || sequence.shape()[1] != 1) throw std::invalid_argument("gru_sequence requires [T,1,I]");
  const std::size_t hidden_size = initial_hidden.shape()[2];
  const std::size_t input_size = sequence.shape()[2];
  std::vector<float> input_weight_values(input_weights.data<float>(), input_weights.data<float>() + 3 * hidden_size * input_size);
  std::vector<float> hidden_weight_values(hidden_weights.data<float>(), hidden_weights.data<float>() + 3 * hidden_size * hidden_size);
  std::vector<float> bias_values(3 * hidden_size);
  for (std::size_t index = 0; index < bias_values.size(); ++index) {
    bias_values[index] = bias.data<float>()[index] + bias.data<float>()[3 * hidden_size + index];
  }
  Tensor flat_input_weights = Tensor::from_vector({3 * hidden_size, input_size}, input_weight_values);
  Tensor flat_hidden_weights = Tensor::from_vector({3 * hidden_size, hidden_size}, hidden_weight_values);
  Tensor flat_bias = Tensor::from_vector({3 * hidden_size}, bias_values);
  Tensor output = pool.allocate_tensor({sequence.shape()[0], 1, hidden_size});
  Tensor hidden = Tensor::from_vector({initial_hidden.shape()[2]},
      std::vector<float>(initial_hidden.data<float>(), initial_hidden.data<float>() + initial_hidden.shape()[2]));
  for (std::size_t time = 0; time < sequence.shape()[0]; ++time) {
    Tensor input = Tensor::from_vector({sequence.shape()[2]},
        std::vector<float>(sequence.data<float>() + time * sequence.shape()[2],
                           sequence.data<float>() + (time + 1) * sequence.shape()[2]));
    Tensor step = cpu::gru_step(input, hidden,
                                flat_input_weights, flat_hidden_weights, flat_bias, &pool);
    std::copy_n(step.data<float>(), hidden.shape()[0], output.data<float>() + time * hidden.shape()[0]);
    hidden = Tensor::from_vector(hidden.shape(), std::vector<float>(
        step.data<float>(), step.data<float>() + hidden.shape()[0]));
  }
  return output;
}

}  // namespace runtime
