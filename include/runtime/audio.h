#pragma once

#include <cstddef>
#include <vector>
#include <string>

namespace runtime {

struct AudioFormat {
  std::size_t sample_rate = 16'000;
  std::size_t channels = 1;
};

class AudioFrame {
 public:
  AudioFrame() = default;
  AudioFrame(AudioFormat format, std::vector<float> samples);

  static AudioFrame mono_16k(std::vector<float> samples);

  const AudioFormat& format() const { return format_; }
  const std::vector<float>& samples() const { return samples_; }
  std::vector<float>& samples() { return samples_; }
  std::size_t sample_count() const { return samples_.size(); }

 private:
  AudioFormat format_;
  std::vector<float> samples_;
};

struct StreamConfig {
  AudioFormat format{};
  std::size_t frame_samples = 160;

  void validate() const;
};

class AudioStream {
 public:
  explicit AudioStream(StreamConfig config);

  const StreamConfig& config() const { return config_; }
  void push(const AudioFrame& frame);
  bool has_frame() const;
  AudioFrame pop();
  void reset();

 private:
  StreamConfig config_;
  std::vector<float> pending_samples_;
};

std::vector<float> read_wav_mono16(const std::string& path, std::size_t expected_sample_rate = 16'000);
void write_wav_mono16(const std::string& path, const std::vector<float>& samples,
                      std::size_t sample_rate = 16'000);

}  // namespace runtime
