#include "runtime/hush_frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "runtime/kernels.h"

namespace runtime {
namespace {
constexpr float kPi = 3.14159265358979323846f;

float freq_to_erb(float frequency) { return 9.265f * std::log1p(frequency / (24.7f * 9.265f)); }
float erb_to_freq(float value) { return 24.7f * 9.265f * (std::exp(value / 9.265f) - 1.0f); }

Tensor make_filterbank(const std::vector<std::size_t>& widths) {
  std::size_t bins = 0;
  for (std::size_t width : widths) bins += width;
  std::vector<float> values(widths.size() * bins, 0.0f);
  std::size_t offset = 0;
  for (std::size_t band = 0; band < widths.size(); ++band) {
    const float scale = 1.0f / static_cast<float>(widths[band]);
    for (std::size_t bin = 0; bin < widths[band]; ++bin) values[band * bins + offset + bin] = scale;
    offset += widths[band];
  }
  return Tensor::from_vector({widths.size(), bins}, values);
}

Tensor make_inverse_filterbank(const std::vector<std::size_t>& widths) {
  std::size_t bins = 0;
  for (std::size_t width : widths) bins += width;
  std::vector<float> values(bins * widths.size(), 0.0f);
  std::size_t offset = 0;
  for (std::size_t band = 0; band < widths.size(); ++band) {
    for (std::size_t bin = 0; bin < widths[band]; ++bin) values[(offset + bin) * widths.size() + band] = 1.0f;
    offset += widths[band];
  }
  return Tensor::from_vector({bins, widths.size()}, values);
}

float vorbis_window(std::size_t index, std::size_t fft_size) {
  const float phase = 0.5f * kPi * (static_cast<float>(index) + 0.5f) / (fft_size / 2.0f);
  const float sine = std::sin(phase);
  return std::sin(0.5f * kPi * sine * sine);
}

Tensor make_hush_spectrum(const std::vector<float>& window, std::size_t fft_size, std::size_t hop_size) {
  const std::size_t bins = fft_size / 2 + 1;
  const float scale = 2.0f * static_cast<float>(hop_size) / static_cast<float>(fft_size * fft_size);
  std::vector<float> values(bins * 2, 0.0f);
  for (std::size_t bin = 0; bin < bins; ++bin) {
    float real = 0.0f, imag = 0.0f;
    for (std::size_t sample = 0; sample < fft_size; ++sample) {
      const float value = window[sample] * vorbis_window(sample, fft_size);
      const float angle = 2.0f * kPi * static_cast<float>(bin * sample) / fft_size;
      real += value * std::cos(angle);
      imag -= value * std::sin(angle);
    }
    values[bin * 2] = real * scale;
    values[bin * 2 + 1] = imag * scale;
  }
  return Tensor::from_vector({1, bins, 2}, values);
}

void reset_normalization_state(std::vector<float>& erb_state, std::vector<float>& df_state) {
  for (std::size_t index = 0; index < erb_state.size(); ++index) {
    erb_state[index] = -60.0f - 30.0f * static_cast<float>(index) /
        static_cast<float>(std::max<std::size_t>(1, erb_state.size() - 1));
  }
  for (std::size_t index = 0; index < df_state.size(); ++index) {
    df_state[index] = 0.001f - 0.0009f * static_cast<float>(index) /
        static_cast<float>(std::max<std::size_t>(1, df_state.size() - 1));
  }
}

void normalize_features(Tensor& erb, Tensor& df, std::vector<float>& erb_state,
                        std::vector<float>& df_state, float alpha) {
  for (std::size_t band = 0; band < erb.shape()[1]; ++band) {
    float& value = erb.data<float>()[band];
    float& state = erb_state[band];
    state = value * (1.0f - alpha) + state * alpha;
    value = (value - state) / 40.0f;
  }
  const std::size_t bins = df.shape()[2];
  for (std::size_t bin = 0; bin < bins; ++bin) {
    const float real = df.data<float>()[bin], imag = df.data<float>()[bins + bin];
    float& state = df_state[bin];
    state = std::sqrt(real * real + imag * imag) * (1.0f - alpha) + state * alpha;
    const float scale = 1.0f / std::sqrt(std::max(state, 1e-12f));
    df.data<float>()[bin] = real * scale;
    df.data<float>()[bins + bin] = imag * scale;
  }
}
}  // namespace

HushFrontend::HushFrontend(const HushConfig& config)
    : config_(config), history_(config.fft_size - config.hop_size, 0.0f),
      erb_widths_(make_erb_widths(config)), filterbank_(make_filterbank(erb_widths_)),
      inverse_filterbank_(make_inverse_filterbank(erb_widths_)), erb_mean_state_(config.erb_bands),
      df_unit_state_(config.df_bins),
      norm_alpha_(std::exp(-static_cast<float>(config.hop_size) /
                           static_cast<float>(config.sample_rate) / config.norm_tau)) {
  config_.validate();
  float precision = 1000.0f;
  for (std::size_t digits = 0; digits < 6; ++digits) {
    const float rounded = std::round(norm_alpha_ * precision) / precision;
    if (rounded < 1.0f) {
      norm_alpha_ = rounded;
      break;
    }
    precision *= 10.0f;
  }
  reset_normalization_state(erb_mean_state_, df_unit_state_);
}

std::vector<std::size_t> HushFrontend::make_erb_widths(const HushConfig& config) {
  const float frequency_width = static_cast<float>(config.sample_rate) / config.fft_size;
  const float low = freq_to_erb(0.0f), high = freq_to_erb(static_cast<float>(config.sample_rate / 2));
  const float step = (high - low) / config.erb_bands;
  std::vector<std::size_t> widths(config.erb_bands, 0);
  int previous_frequency = 0, carried = 0;
  for (std::size_t index = 1; index <= config.erb_bands; ++index) {
    const int frequency_bin = static_cast<int>(std::lround(erb_to_freq(low + index * step) / frequency_width));
    int width = frequency_bin - previous_frequency - carried;
    if (width < static_cast<int>(config.min_nb_erb_freqs)) {
      carried = static_cast<int>(config.min_nb_erb_freqs) - width;
      width = static_cast<int>(config.min_nb_erb_freqs);
    } else {
      carried = 0;
    }
    widths[index - 1] = static_cast<std::size_t>(width);
    previous_frequency = frequency_bin;
  }
  ++widths.back();
  std::size_t total = 0;
  for (std::size_t width : widths) total += width;
  if (total > config.fft_size / 2 + 1) widths.back() -= total - (config.fft_size / 2 + 1);
  return widths;
}

HushFeatures HushFrontend::process(const AudioFrame& frame) {
  if (frame.format().sample_rate != config_.sample_rate || frame.format().channels != 1 ||
      frame.sample_count() != config_.hop_size) throw std::invalid_argument("audio frame does not match Hush frontend configuration");
  std::vector<float> window(config_.fft_size, 0.0f);
  std::copy(history_.begin(), history_.end(), window.begin());
  std::copy(frame.samples().begin(), frame.samples().end(), window.begin() + history_.size());
  Tensor spectrum = make_hush_spectrum(window, config_.fft_size, config_.hop_size);
  Tensor erb = Tensor::empty({1, config_.erb_bands});
  std::size_t offset = 0;
  for (std::size_t band = 0; band < erb_widths_.size(); ++band) {
    float energy = 0.0f;
    for (std::size_t bin = 0; bin < erb_widths_[band]; ++bin) {
      const float real = spectrum.data<float>()[(offset + bin) * 2];
      const float imag = spectrum.data<float>()[(offset + bin) * 2 + 1];
      energy += real * real + imag * imag;
    }
    erb.data<float>()[band] = 10.0f * std::log10(energy / erb_widths_[band] + 1e-10f);
    offset += erb_widths_[band];
  }
  const std::size_t df_bins = std::min(config_.df_bins, spectrum.shape()[1]);
  Tensor df = Tensor::empty({2, 1, df_bins});
  for (std::size_t bin = 0; bin < df_bins; ++bin) {
    df.data<float>()[bin] = spectrum.data<float>()[bin * 2];
    df.data<float>()[df_bins + bin] = spectrum.data<float>()[bin * 2 + 1];
  }
  normalize_features(erb, df, erb_mean_state_, df_unit_state_, norm_alpha_);
  history_ = frame.samples();
  return {.spectrum = std::move(spectrum), .erb = std::move(erb), .df = std::move(df)};
}

HushFeatures HushFrontend::process(const AudioFrame& frame, const Backend& backend, MemoryPool& pool) {
  (void)backend;
  (void)pool;
  return process(frame);
}

void HushFrontend::reset() {
  std::fill(history_.begin(), history_.end(), 0.0f);
  reset_normalization_state(erb_mean_state_, df_unit_state_);
}
}  // namespace runtime
