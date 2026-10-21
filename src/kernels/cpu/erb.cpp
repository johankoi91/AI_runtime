#include "runtime/kernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace runtime::cpu {

Tensor erb_filterbank(
    const Tensor& spectrum,
    const Tensor& filterbank,
    MemoryPool* pool) {
  if (spectrum.dtype() != DType::Float32 || filterbank.dtype() != DType::Float32 ||
      !spectrum.is_contiguous() || !filterbank.is_contiguous() || spectrum.rank() != 3 ||
      filterbank.rank() != 2 || spectrum.shape()[2] != 2 ||
      filterbank.shape()[1] != spectrum.shape()[1]) {
    throw std::invalid_argument("erb_filterbank requires spectrum [frames,bins,2] and filterbank [bands,bins]");
  }
  const std::size_t frames = spectrum.shape()[0];
  const std::size_t bins = spectrum.shape()[1];
  const std::size_t bands = filterbank.shape()[0];
  Tensor output = pool == nullptr
      ? Tensor::empty({frames, bands})
      : pool->allocate_tensor({frames, bands});
  const float* spectrum_data = spectrum.data<float>();
  const float* filterbank_data = filterbank.data<float>();
  float* output_data = output.data<float>();
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (std::size_t band = 0; band < bands; ++band) {
      float energy = 0.0f;
      for (std::size_t bin = 0; bin < bins; ++bin) {
        const std::size_t spectrum_offset = (frame * bins + bin) * 2;
        const float real = spectrum_data[spectrum_offset];
        const float imag = spectrum_data[spectrum_offset + 1];
        energy += filterbank_data[band * bins + bin] * (real * real + imag * imag);
      }
      output_data[frame * bands + band] = std::log(std::max(energy, 1e-12f));
    }
  }
  return output;
}

Tensor exponential_unit_norm(
    const Tensor& input,
    float& running_energy,
    float alpha,
    MemoryPool* pool) {
  if (input.dtype() != DType::Float32 || !input.is_contiguous() || input.rank() != 1 ||
      !(alpha > 0.0f && alpha <= 1.0f)) {
    throw std::invalid_argument("exponential_unit_norm requires a contiguous vector and 0 < alpha <= 1");
  }
  float energy = 0.0f;
  for (std::size_t index = 0; index < input.numel(); ++index) {
    const float value = input.data<float>()[index];
    energy += value * value;
  }
  energy /= static_cast<float>(std::max<std::size_t>(input.numel(), 1));
  running_energy = (1.0f - alpha) * running_energy + alpha * energy;
  Tensor output = pool == nullptr
      ? Tensor::empty(input.shape(), input.dtype())
      : pool->allocate_tensor(input.shape(), input.dtype());
  const float scale = 1.0f / std::sqrt(std::max(running_energy, 1e-12f));
  for (std::size_t index = 0; index < input.numel(); ++index) {
    output.data<float>()[index] = input.data<float>()[index] * scale;
  }
  return output;
}

}  // namespace runtime::cpu
