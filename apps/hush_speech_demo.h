#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/audio.h"

struct HushSpeechDemo {
  std::vector<float> target;
  std::vector<float> background;
  std::vector<float> mixed;
};

inline HushSpeechDemo make_hush_speech_demo(
    const std::string& target_path, const std::string& background_path, float input_sir_db) {
  HushSpeechDemo demo;
  demo.target = runtime::read_wav_mono16(target_path);
  demo.background = runtime::read_wav_mono16(background_path);
  if (demo.target.empty() || demo.background.empty() || !std::isfinite(input_sir_db)) {
    throw std::invalid_argument("speech demo requires two nonempty recordings and a finite SIR");
  }
  const std::size_t length = std::max(demo.target.size(), demo.background.size());
  demo.target.resize(length, 0.0f);
  demo.background.resize(length, 0.0f);
  auto normalize = [](std::vector<float>& samples, double level) {
    double energy = 0.0;
    for (float value : samples) energy += static_cast<double>(value) * value;
    if (energy <= 0.0) throw std::invalid_argument("speech demo source is silent");
    const float gain = static_cast<float>(level / std::sqrt(energy / samples.size()));
    for (float& value : samples) value *= gain;
  };
  normalize(demo.target, 0.04);
  normalize(demo.background, 0.04 * std::pow(10.0, -input_sir_db / 20.0));
  demo.mixed.resize(length);
  float peak = 0.0f;
  for (std::size_t index = 0; index < length; ++index) {
    demo.mixed[index] = demo.target[index] + demo.background[index];
    peak = std::max(peak, std::fabs(demo.mixed[index]));
  }
  if (peak > 0.95f) {
    const float gain = 0.95f / peak;
    for (std::size_t index = 0; index < length; ++index) {
      demo.target[index] *= gain;
      demo.background[index] *= gain;
      demo.mixed[index] *= gain;
    }
  }
  return demo;
}
