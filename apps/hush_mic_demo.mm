#import <AVFoundation/AVFoundation.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "runtime/hush_model.h"
#include "runtime/hush_session.h"
#include "runtime/metal_backend.h"

int main() {
  const char* bundle_path = std::getenv("HUSH_MODEL_BUNDLE");
  if (bundle_path == nullptr || *bundle_path == '\0') { std::cerr << "HUSH_MODEL_BUNDLE is required\n"; return 2; }
  @autoreleasepool {
    NSError* error = nil;
    AVAudioEngine* engine = [[AVAudioEngine alloc] init];
    AVAudioInputNode* input = engine.inputNode;
    AVAudioFormat* format = [[AVAudioFormat alloc] initStandardFormatWithSampleRate:16000 channels:1];
    if (![input setVoiceProcessingEnabled:NO error:&error]) {
      std::cerr << "audio input setup failed: " << error.localizedDescription.UTF8String << "\n";
      return 1;
    }
    const runtime::HushModel model = runtime::HushModel::load(bundle_path);
    runtime::MetalBackend backend;
    runtime::HushStreamSession session(model, backend, {});
    runtime::MemoryPool pool;
    __block std::vector<float> pending;
    runtime::HushStreamSession* session_ptr = &session;
    runtime::MemoryPool* pool_ptr = &pool;
    runtime::MetalBackend* backend_ptr = &backend;
    [input installTapOnBus:0 bufferSize:160 format:format block:^(AVAudioPCMBuffer* buffer, AVAudioTime*) {
      const float* samples = buffer.floatChannelData[0];
      pending.insert(pending.end(), samples, samples + buffer.frameLength);
      while (pending.size() >= 160) {
        std::vector<float> frame(pending.begin(), pending.begin() + 160);
        pending.erase(pending.begin(), pending.begin() + 160);
        try {
          (void)session_ptr->process_audio(runtime::AudioFrame::mono_16k(std::move(frame)),
                                            "feat_erb", "feat_spec", "m", *backend_ptr, *pool_ptr);
          pool_ptr->reset();
        } catch (const std::exception& exception) {
          std::cerr << "Hush frame failed: " << exception.what() << "\n";
        }
      }
    }];
    [engine prepare];
    if (![engine startAndReturnError:&error]) { std::cerr << "audio engine failed: " << error.localizedDescription.UTF8String << "\n"; return 1; }
    std::cout << "Hush Metal microphone demo running; press Enter to stop\n";
    std::cin.get();
    [input removeTapOnBus:0];
    [engine stop];
  }
  return 0;
}
