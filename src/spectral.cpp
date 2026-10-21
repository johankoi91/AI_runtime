#include "runtime/spectral.h"

#include <cmath>
#include <complex>
#include <stdexcept>

namespace runtime {

namespace {

constexpr float kPi = 3.14159265358979323846f;

float hann_window(std::size_t index, std::size_t size) {
  return 0.5f * (1.0f - std::cos(2.0f * kPi * static_cast<float>(index) /
                                  static_cast<float>(size)));
}

std::size_t frame_count(std::size_t sample_count, std::size_t fft_size, std::size_t hop_size) {
  if (sample_count <= fft_size) {
    return 1;
  }
  return (sample_count - fft_size + hop_size - 1) / hop_size + 1;
}

}  // namespace

const Complex32& Spectrogram::at(std::size_t frame, std::size_t bin) const {
  if (bins == 0 || frame >= values.size() / bins || bin >= bins) {
    throw std::out_of_range("spectrogram index is out of range");
  }
  return values.at(frame * bins + bin);
}

Complex32& Spectrogram::at(std::size_t frame, std::size_t bin) {
  if (bins == 0 || frame >= values.size() / bins || bin >= bins) {
    throw std::out_of_range("spectrogram index is out of range");
  }
  return values.at(frame * bins + bin);
}

Spectrogram stft(
    const std::vector<float>& samples,
    std::size_t fft_size,
    std::size_t hop_size) {
  if (fft_size == 0 || hop_size == 0 || hop_size > fft_size) {
    throw std::invalid_argument("stft requires fft_size >= hop_size > 0");
  }

  const std::size_t frames = frame_count(samples.size(), fft_size, hop_size);
  const std::size_t bins = fft_size / 2 + 1;
  Spectrogram result{.fft_size = fft_size,
                     .hop_size = hop_size,
                     .original_samples = samples.size(),
                     .bins = bins,
                     .values = std::vector<Complex32>(frames * bins)};

  for (std::size_t frame = 0; frame < frames; ++frame) {
    const std::size_t frame_start = frame * hop_size;
    for (std::size_t bin = 0; bin < bins; ++bin) {
      float real_sum = 0.0f;
      float imag_sum = 0.0f;
      for (std::size_t sample_index = 0; sample_index < fft_size; ++sample_index) {
        const std::size_t input_index = frame_start + sample_index;
        const float sample = input_index < samples.size() ? samples[input_index] : 0.0f;
        const float windowed_sample = sample * hann_window(sample_index, fft_size);
        const float angle = -2.0f * kPi * static_cast<float>(bin * sample_index) /
            static_cast<float>(fft_size);
        real_sum += windowed_sample * std::cos(angle);
        imag_sum += windowed_sample * std::sin(angle);
      }
      result.at(frame, bin) = {.real = real_sum, .imag = imag_sum};
    }
  }
  return result;
}

std::vector<float> istft(const Spectrogram& spectrogram) {
  if (spectrogram.fft_size == 0 || spectrogram.hop_size == 0 ||
      spectrogram.hop_size > spectrogram.fft_size ||
      spectrogram.bins != spectrogram.fft_size / 2 + 1) {
    throw std::invalid_argument("invalid spectrogram metadata");
  }
  const std::size_t frames = spectrogram.values.size() / spectrogram.bins;
  std::vector<float> output(spectrogram.original_samples + spectrogram.fft_size, 0.0f);
  std::vector<float> window_energy(output.size(), 0.0f);

  for (std::size_t frame = 0; frame < frames; ++frame) {
    const std::size_t frame_start = frame * spectrogram.hop_size;
    for (std::size_t sample_index = 0; sample_index < spectrogram.fft_size; ++sample_index) {
      float real_sum = 0.0f;
      for (std::size_t bin = 0; bin < spectrogram.fft_size; ++bin) {
        Complex32 frequency{};
        if (bin < spectrogram.bins) {
          frequency = spectrogram.at(frame, bin);
        } else {
          const std::size_t mirror = spectrogram.fft_size - bin;
          if (mirror == 0 || mirror >= spectrogram.bins - 1) {
            continue;
          }
          const Complex32 source = spectrogram.at(frame, mirror);
          frequency = {.real = source.real, .imag = -source.imag};
        }
        const float angle = 2.0f * kPi * static_cast<float>(bin * sample_index) /
            static_cast<float>(spectrogram.fft_size);
        real_sum += frequency.real * std::cos(angle) - frequency.imag * std::sin(angle);
      }
      const float window = hann_window(sample_index, spectrogram.fft_size);
      const std::size_t output_index = frame_start + sample_index;
      output[output_index] += real_sum / static_cast<float>(spectrogram.fft_size) * window;
      window_energy[output_index] += window * window;
    }
  }

  output.resize(spectrogram.original_samples);
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (window_energy[index] > 1e-8f) {
      output[index] /= window_energy[index];
    } else {
      output[index] = 0.0f;
    }
  }
  return output;
}

}  // namespace runtime
