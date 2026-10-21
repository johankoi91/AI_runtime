#pragma once

#include <cstddef>
#include <vector>

namespace runtime {

struct Complex32 {
  float real = 0.0f;
  float imag = 0.0f;
};

struct Spectrogram {
  std::size_t fft_size = 0;
  std::size_t hop_size = 0;
  std::size_t original_samples = 0;
  std::size_t bins = 0;
  std::vector<Complex32> values;

  const Complex32& at(std::size_t frame, std::size_t bin) const;
  Complex32& at(std::size_t frame, std::size_t bin);
};

Spectrogram stft(
    const std::vector<float>& samples,
    std::size_t fft_size = 320,
    std::size_t hop_size = 160);

std::vector<float> istft(const Spectrogram& spectrogram);

}  // namespace runtime
