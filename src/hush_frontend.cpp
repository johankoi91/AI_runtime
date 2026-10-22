#include "runtime/hush_frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "runtime/kernels.h"
#include "runtime/spectral.h"

namespace runtime {

namespace {

Tensor make_filterbank(std::size_t bands, std::size_t bins) {
  std::vector<float> values(bands * bins, 0.0f);
  for (std::size_t band = 0; band < bands; ++band) {
    const float left = static_cast<float>(band) * static_cast<float>(bins - 1) /
        static_cast<float>(bands);
    const float center = static_cast<float>(band + 0.5f) * static_cast<float>(bins - 1) /
        static_cast<float>(bands);
    const float right = static_cast<float>(band + 1) * static_cast<float>(bins - 1) /
        static_cast<float>(bands);
    for (std::size_t bin = 0; bin < bins; ++bin) {
      const float position = static_cast<float>(bin);
      const float rising = center > left ? (position - left) / (center - left) : 0.0f;
      const float falling = right > center ? (right - position) / (right - center) : 0.0f;
      values[band * bins + bin] = std::max(0.0f, std::min(rising, falling));
    }
  }
  return Tensor::from_vector({bands, bins}, values);
}

}  // namespace

HushFrontend::HushFrontend(const HushConfig& config)
    : config_(config), history_(config.hop_size, 0.0f),
      filterbank_(make_filterbank(config.erb_bands, config.fft_size / 2 + 1)) {
  config_.validate();
}

HushFeatures HushFrontend::process(const AudioFrame& frame) {
  if (frame.format().sample_rate != config_.sample_rate || frame.format().channels != 1 ||
      frame.sample_count() != config_.hop_size) {
    throw std::invalid_argument("audio frame does not match Hush frontend configuration");
  }
  std::vector<float> window;
  window.reserve(config_.fft_size);
  window.insert(window.end(), history_.begin(), history_.end());
  window.insert(window.end(), frame.samples().begin(), frame.samples().end());
  const Spectrogram spectrogram = stft(window, config_.fft_size, config_.hop_size);
  std::vector<float> spectrum_values;
  spectrum_values.reserve(spectrogram.values.size() * 2);
  for (const Complex32& value : spectrogram.values) {
    spectrum_values.push_back(value.real);
    spectrum_values.push_back(value.imag);
  }
  Tensor spectrum = Tensor::from_vector(
      {spectrogram.values.size() / spectrogram.bins, spectrogram.bins, 2}, spectrum_values);
  Tensor erb = cpu::erb_filterbank(spectrum, filterbank_);
  history_ = frame.samples();
  return {.spectrum = std::move(spectrum), .erb = std::move(erb)};
}

void HushFrontend::reset() {
  std::fill(history_.begin(), history_.end(), 0.0f);
}

}  // namespace runtime
