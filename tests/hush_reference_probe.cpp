#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "runtime/cpu_backend.h"
#include "runtime/metal_backend.h"
#include "runtime/hush_session.h"

void dump(const std::string& directory, const std::string& graph,
          const std::unordered_map<std::string, runtime::Tensor>& tensors) {
  std::ofstream manifest(directory + "/" + graph + ".tsv");
  std::size_t index = 0;
  for (const auto& item : tensors) {
    const std::string path = graph + "-" + std::to_string(index++) + ".f32";
    std::ofstream file(directory + "/" + path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(item.second.data<float>()), item.second.nbytes());
    manifest << item.first << '\t' << path << '\t';
    for (auto size : item.second.shape()) manifest << size << ',';
    manifest << '\n';
  }
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    const char* bundle_path = std::getenv("HUSH_MODEL_BUNDLE");
    if (bundle_path == nullptr) throw std::runtime_error("HUSH_MODEL_BUNDLE is required");
    std::filesystem::create_directories(argv[2]);
    const auto samples = runtime::read_wav_mono16(argv[1]);
    const auto model = runtime::HushModel::load(bundle_path);
    runtime::HushFrontend frontend(model.config);
    const std::size_t count = (samples.size() + 159) / 160;
    if (count < 32) throw std::runtime_error("reference input must contain at least 32 frames");
    std::vector<float> erb(count * 32), df(2 * count * 64), spectrum(count * 161 * 2);
    for (std::size_t time = 0; time < count; ++time) {
      std::vector<float> frame(160, 0.0f);
      std::copy_n(samples.data() + time * 160, std::min<std::size_t>(160, samples.size() - time * 160), frame.data());
      const auto features = frontend.process(runtime::AudioFrame::mono_16k(frame));
      std::copy_n(features.erb.data<float>(), 32, erb.data() + time * 32);
      std::copy_n(features.df.data<float>(), 64, df.data() + time * 64);
      std::copy_n(features.df.data<float>() + 64, 64, df.data() + (count + time) * 64);
      std::copy_n(features.spectrum.data<float>(), 322, spectrum.data() + time * 322);
    }
    dump(argv[2], "features", {{"feat_erb", runtime::Tensor::from_vector({1,1,count,32}, erb)},
          {"feat_spec", runtime::Tensor::from_vector({1,2,count,64}, df)},
          {"spectrum", runtime::Tensor::from_vector({count,161,2}, spectrum)}});
    runtime::CPUBackend cpu;
    runtime::MetalBackend metal;
    const runtime::Backend& backend = std::getenv("HUSH_BACKEND") == nullptr ? static_cast<const runtime::Backend&>(cpu) : metal;
    runtime::HushStreamSession session(model, backend);
    runtime::MemoryPool pool;
    std::vector<float> first_erb(32*32), first_df(2*32*64);
    std::copy_n(erb.data(), first_erb.size(), first_erb.data());
    std::copy_n(df.data(), 32*64, first_df.data());
    std::copy_n(df.data() + count*64, 32*64, first_df.data() + 32*64);
    const auto encoder = session.process_encoder({{"feat_erb", runtime::Tensor::from_vector({1,1,32,32}, first_erb)},
        {"feat_spec", runtime::Tensor::from_vector({1,2,32,64}, first_df)}}, pool);
    dump(argv[2], "enc", encoder);
    const auto decoders = session.process_decoders(encoder, {}, {}, pool);
    dump(argv[2], "erb", decoders.erb);
    dump(argv[2], "df", decoders.df);
    pool.reset();
    session.reset();
    std::vector<float> enhanced;
    const char* batch_text = std::getenv("HUSH_TEST_BATCH_SIZE");
    const std::size_t batch_size = batch_text == nullptr ? 32 : std::stoul(batch_text);
    if (batch_size == 0 || batch_size > 32) throw std::runtime_error("batch size must be 1..32");
    for (std::size_t offset = 0; offset < count; offset += batch_size) {
      std::vector<runtime::AudioFrame> frames;
      for (std::size_t time = offset; time < std::min(count, offset + batch_size); ++time) {
        std::vector<float> frame(160, 0.0f);
        std::copy_n(samples.data() + time * 160, std::min<std::size_t>(160, samples.size() - time * 160), frame.data());
        frames.push_back(runtime::AudioFrame::mono_16k(std::move(frame)));
      }
      const auto outputs = session.process_audio_batch(frames, "feat_erb", "feat_spec", "m", backend, pool);
      for (const auto& output : outputs) enhanced.insert(enhanced.end(), output.samples().begin(), output.samples().end());
      pool.reset();
    }
    enhanced.resize(samples.size());
    dump(argv[2], "audio", {{"output", runtime::Tensor::from_vector({enhanced.size()}, enhanced)}});
    runtime::write_wav_mono16(std::string(argv[2]) + "/output.wav", enhanced);
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
