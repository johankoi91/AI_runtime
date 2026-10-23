#import <AVFoundation/AVFoundation.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "runtime/audio.h"
#include "runtime/cpu_backend.h"
#include "runtime/hush_model.h"
#include "runtime/hush_session.h"
#include "runtime/metal_backend.h"

namespace {
constexpr double kPi = 3.14159265358979323846;

std::vector<float> make_voice(double frequency, double modulation, std::size_t samples, float gain) {
  std::vector<float> output(samples);
  for (std::size_t index = 0; index < samples; ++index) {
    const double time = static_cast<double>(index) / 16'000.0;
    const double envelope = 0.65 + 0.35 * std::sin(2.0 * kPi * 2.1 * time);
    const double fundamental = std::sin(2.0 * kPi * frequency * time + modulation * std::sin(2.0 * kPi * 3.0 * time));
    const double harmonic = 0.35 * std::sin(2.0 * kPi * frequency * 2.01 * time);
    output[index] = gain * static_cast<float>(envelope * (fundamental + harmonic));
  }
  return output;
}

void play_wav(const std::string& path) {
  AVAudioPlayer* player = [[AVAudioPlayer alloc] initWithContentsOfURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]] error:nil];
  [player play];
  while (player.isPlaying) [[NSRunLoop currentRunLoop] runUntilDate:
      [NSDate dateWithTimeIntervalSinceNow:0.05]];
}
}

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    const char* bundle = std::getenv("HUSH_MODEL_BUNDLE");
    if (bundle == nullptr || *bundle == '\0') { std::cerr << "HUSH_MODEL_BUNDLE is required\n"; return 2; }
    const bool play = argc > 1 && std::string(argv[1]) == "--play";
    try {
      const std::size_t samples_count = 16'000 * 3;
      const auto primary = make_voice(180.0, 0.7, samples_count, 0.18f);
      const auto background = make_voice(265.0, 1.1, samples_count, 0.13f);
      std::vector<float> mixed(samples_count);
      for (std::size_t index = 0; index < samples_count; ++index) mixed[index] = primary[index] + background[index];
      runtime::write_wav_mono16("/tmp/hush-background-input.wav", mixed);

      const runtime::HushModel model = runtime::HushModel::load(bundle);
      runtime::MetalBackend backend;
      runtime::HushStreamSession session(model, backend, {});
      runtime::MemoryPool pool;
      std::vector<float> enhanced;
      enhanced.reserve(samples_count);
      for (std::size_t offset = 0; offset < samples_count; offset += 160) {
        std::vector<float> frame(mixed.begin() + offset, mixed.begin() + offset + 160);
        const auto output = session.process_audio(
            runtime::AudioFrame::mono_16k(std::move(frame)),
            "feat_erb", "feat_spec", "m", backend, pool);
        enhanced.insert(enhanced.end(), output.samples().begin(), output.samples().end());
        pool.reset();
      }
      runtime::write_wav_mono16("/tmp/hush-background-output.wav", enhanced);
      std::cout << "input=/tmp/hush-background-input.wav\noutput=/tmp/hush-background-output.wav\n"
                << "description=primary voice plus simulated background speaker -> Hush Metal output\n";
      if (play) {
        std::cout << "playing mixed input...\n";
        play_wav("/tmp/hush-background-input.wav");
        std::cout << "playing enhanced output...\n";
        play_wav("/tmp/hush-background-output.wav");
      }
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "interference demo failed: " << error.what() << "\n";
      return 1;
    }
  }
}
