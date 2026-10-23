#import <AVFoundation/AVFoundation.h>

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
#include "hush_speech_demo.h"

namespace {
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
      const auto demo = make_hush_speech_demo(
          std::string(HUSH_DEMO_AUDIO_DIR) + "/hush-demo-target.wav",
          std::string(HUSH_DEMO_AUDIO_DIR) + "/hush-demo-background.wav", 18.0f);
      const auto& mixed = demo.mixed;
      const std::size_t samples_count = mixed.size();
      runtime::write_wav_mono16("/tmp/hush-background-target.wav", demo.target);
      runtime::write_wav_mono16("/tmp/hush-background-interferer.wav", demo.background);
      runtime::write_wav_mono16("/tmp/hush-background-input.wav", mixed);

      const runtime::HushModel model = runtime::HushModel::load(bundle);
      runtime::MetalBackend backend;
      runtime::HushStreamSession session(model, backend, {});
      runtime::MemoryPool pool;
      std::vector<float> enhanced;
      enhanced.reserve(samples_count);
      for (std::size_t offset = 0; offset < samples_count; offset += 160) {
        std::vector<float> frame(160, 0.0f);
        std::copy_n(mixed.data() + offset, std::min<std::size_t>(160, samples_count - offset), frame.data());
        const auto output = session.process_audio(
            runtime::AudioFrame::mono_16k(std::move(frame)),
            "feat_erb", "feat_spec", "m", backend, pool);
        enhanced.insert(enhanced.end(), output.samples().begin(), output.samples().end());
        pool.reset();
      }
      enhanced.resize(samples_count);
      runtime::write_wav_mono16("/tmp/hush-background-output.wav", enhanced);
      std::cout << "input=/tmp/hush-background-input.wav\noutput=/tmp/hush-background-output.wav\n"
                << "target=/tmp/hush-background-target.wav\n"
                << "description=two synthesized Chinese voices, background 18 dB lower -> Hush Metal output\n";
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
