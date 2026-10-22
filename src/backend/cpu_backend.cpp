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
    std::size_t padding,
    std::size_t groups,
    MemoryPool& pool) const {
  return cpu::conv2d(input, weights, bias, stride_h, stride_w, padding, groups, &pool);
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

}  // namespace runtime
