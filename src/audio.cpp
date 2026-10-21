#include "runtime/audio.h"

#include <algorithm>
#include <stdexcept>

namespace runtime {

AudioFrame::AudioFrame(AudioFormat format, std::vector<float> samples)
    : format_(format), samples_(std::move(samples)) {
  if (format_.sample_rate == 0 || format_.channels == 0) {
    throw std::invalid_argument("audio frame format must have non-zero rate and channels");
  }
  if (samples_.size() % format_.channels != 0) {
    throw std::invalid_argument("audio frame samples must be interleaved by channel");
  }
}

AudioFrame AudioFrame::mono_16k(std::vector<float> samples) {
  return AudioFrame(AudioFormat{16'000, 1}, std::move(samples));
}

void StreamConfig::validate() const {
  if (format.sample_rate == 0 || format.channels == 0 || frame_samples == 0) {
    throw std::invalid_argument("stream config must have non-zero format and frame size");
  }
}

AudioStream::AudioStream(StreamConfig config) : config_(config) {
  config_.validate();
}

void AudioStream::push(const AudioFrame& frame) {
  if (frame.format().sample_rate != config_.format.sample_rate ||
      frame.format().channels != config_.format.channels) {
    throw std::invalid_argument("audio frame format does not match stream config");
  }
  pending_samples_.insert(
      pending_samples_.end(), frame.samples().begin(), frame.samples().end());
}

bool AudioStream::has_frame() const {
  return pending_samples_.size() >= config_.frame_samples * config_.format.channels;
}

AudioFrame AudioStream::pop() {
  if (!has_frame()) {
    throw std::runtime_error("audio stream does not contain a complete frame");
  }

  const std::size_t frame_size = config_.frame_samples * config_.format.channels;
  std::vector<float> samples(pending_samples_.begin(), pending_samples_.begin() + frame_size);
  pending_samples_.erase(pending_samples_.begin(), pending_samples_.begin() + frame_size);
  return AudioFrame(config_.format, std::move(samples));
}

void AudioStream::reset() {
  pending_samples_.clear();
}

}  // namespace runtime
