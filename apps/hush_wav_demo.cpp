#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "runtime/audio.h"
#include "runtime/cpu_backend.h"
#include "runtime/hush_model.h"
#include "runtime/hush_session.h"
#include "runtime/metal_backend.h"

int main(int argc, char** argv) {
  if (argc != 3) { std::cerr << "usage: hush_wav_demo input.wav output.wav\n"; return 2; }
  const char* bundle = std::getenv("HUSH_MODEL_BUNDLE"); if (bundle == nullptr) return 2;
  try {
    const auto samples = runtime::read_wav_mono16(argv[1]);
    const runtime::HushModel model = runtime::HushModel::load(bundle);
    std::unique_ptr<runtime::Backend> backend;
    if (const char* name = std::getenv("HUSH_BACKEND"); name != nullptr && std::string(name) == "metal") backend = std::make_unique<runtime::MetalBackend>();
    else backend = std::make_unique<runtime::CPUBackend>();
    runtime::HushStreamSession session(model, *backend, {}); runtime::MemoryPool pool;
    std::vector<float> output; output.reserve(samples.size());
    for (std::size_t offset = 0; offset < samples.size(); offset += 160) {
      std::vector<float> frame(160, 0.0f); const std::size_t count = std::min<std::size_t>(160, samples.size() - offset); std::copy_n(samples.data() + offset, count, frame.data());
      const auto enhanced = session.process_audio(runtime::AudioFrame::mono_16k(frame), "feat_erb", "feat_spec", "m", *backend, pool);
      output.insert(output.end(), enhanced.samples().begin(), enhanced.samples().end()); pool.reset();
    }
    output.resize(samples.size()); runtime::write_wav_mono16(argv[2], output); std::cout << "input_samples=" << samples.size() << " output_samples=" << output.size() << " backend=" << backend->name() << "\n"; return 0;
  } catch (const std::exception& error) { std::cerr << "WAV demo failed: " << error.what() << "\n"; return 1; }
}
