#include "runtime/kernels.h"

#include <stdexcept>

namespace runtime::cpu {

Tensor apply_erb_mask(
    const Tensor& spectrum,
    const Tensor& erb_mask,
    const Tensor& erb_projection,
    MemoryPool* pool) {
  if (spectrum.dtype() != DType::Float32 || erb_mask.dtype() != DType::Float32 ||
      erb_projection.dtype() != DType::Float32 || !spectrum.is_contiguous() ||
      !erb_mask.is_contiguous() || !erb_projection.is_contiguous() || spectrum.rank() != 3 ||
      erb_mask.rank() != 2 || erb_projection.rank() != 2 || spectrum.shape()[2] != 2 ||
      erb_mask.shape()[0] != spectrum.shape()[0] ||
      erb_projection.shape() != Shape({spectrum.shape()[1], erb_mask.shape()[1]})) {
    throw std::invalid_argument("apply_erb_mask has invalid tensor shapes");
  }
  Tensor output = pool == nullptr ? Tensor::empty(spectrum.shape()) : pool->allocate_tensor(spectrum.shape());
  const std::size_t frames = spectrum.shape()[0];
  const std::size_t bins = spectrum.shape()[1];
  const std::size_t bands = erb_mask.shape()[1];
  for (std::size_t frame = 0; frame < frames; ++frame) {
    for (std::size_t bin = 0; bin < bins; ++bin) {
      float gain = 0.0f;
      for (std::size_t band = 0; band < bands; ++band) {
        gain += erb_mask.data<float>()[frame * bands + band] *
            erb_projection.data<float>()[bin * bands + band];
      }
      const std::size_t offset = (frame * bins + bin) * 2;
      output.data<float>()[offset] = spectrum.data<float>()[offset] * gain;
      output.data<float>()[offset + 1] = spectrum.data<float>()[offset + 1] * gain;
    }
  }
  return output;
}

}  // namespace runtime::cpu
