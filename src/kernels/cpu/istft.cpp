#include "runtime/kernels.h"

#include <stdexcept>
#include <vector>

#include "runtime/spectral.h"

namespace runtime::cpu {

Tensor istft(
    const Tensor& spectrum,
    std::size_t fft_size,
    std::size_t hop_size,
    std::size_t original_samples,
    MemoryPool* pool) {
  if (spectrum.dtype() != DType::Float32 || !spectrum.is_contiguous() || spectrum.rank() != 3 ||
      spectrum.shape()[2] != 2 || spectrum.shape()[1] != fft_size / 2 + 1) {
    throw std::invalid_argument("istft requires a contiguous [frames,bins,2] spectrum");
  }
  Spectrogram input{.fft_size = fft_size,
                    .hop_size = hop_size,
                    .original_samples = original_samples,
                    .bins = spectrum.shape()[1],
                    .values = std::vector<Complex32>(spectrum.shape()[0] * spectrum.shape()[1])};
  for (std::size_t index = 0; index < input.values.size(); ++index) {
    input.values[index] = {spectrum.data<float>()[index * 2], spectrum.data<float>()[index * 2 + 1]};
  }
  const std::vector<float> values = runtime::istft(input);
  Tensor output = pool == nullptr ? Tensor::empty({values.size()}) : pool->allocate_tensor({values.size()});
  std::copy(values.begin(), values.end(), output.data<float>());
  return output;
}

}  // namespace runtime::cpu
