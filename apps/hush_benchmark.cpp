#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>

#include "runtime/cpu_backend.h"
#include "runtime/hush_frontend.h"
#include "runtime/hush_model.h"
#include "runtime/hush_session.h"
#include "runtime/metal_backend.h"

int main() {
  const char* path = std::getenv("HUSH_MODEL_BUNDLE");
  if (path == nullptr || *path == '\0') return 2;
  try {
    const runtime::HushModel model = runtime::HushModel::load(path);
    std::unique_ptr<runtime::Backend> backend;
    const char* backend_name = std::getenv("HUSH_BACKEND");
    if (backend_name != nullptr && std::string(backend_name) == "metal") {
      backend = std::make_unique<runtime::MetalBackend>();
    } else {
      backend = std::make_unique<runtime::CPUBackend>();
    }
    runtime::HushFrontend frontend(model.config);
    std::vector<float> erb_values(32 * 32);
    std::vector<float> df_values(2 * 32 * 64);
    for (int time = 0; time < 32; ++time) {
      std::vector<float> samples(160);
      for (std::size_t index = 0; index < samples.size(); ++index) {
        samples[index] = 0.05f * std::sin(static_cast<float>(index + time * 160) * 0.07f);
      }
      const auto features = frontend.process(runtime::AudioFrame::mono_16k(samples));
      std::copy_n(features.erb.data<float>(), 32, erb_values.data() + time * 32);
      for (std::size_t channel = 0; channel < 2; ++channel) {
        std::copy_n(features.df.data<float>() + channel * 64, 64,
                    df_values.data() + (channel * 32 + time) * 64);
      }
    }
    runtime::Tensor erb = runtime::Tensor::from_vector({1, 1, 32, 32}, erb_values);
    runtime::Tensor df = runtime::Tensor::from_vector({1, 2, 32, 64}, df_values);
    const int iterations = 5;
    double total_ms = 0.0;
    std::vector<double> durations;
    double encoder_ms = 0.0, decoder_ms = 0.0, audio_ms = 0.0;
    std::size_t output_count = 0;
    runtime::HushStreamSession session(model, *backend, {});
    runtime::MemoryPool pool;
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      const auto encoder_start = start;
      const auto encoder = session.process_encoder({{"feat_erb", erb}, {"feat_spec", df}}, pool);
      encoder_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - encoder_start).count();
      const auto decoder_start = std::chrono::steady_clock::now();
      const auto decoders = session.process_decoders(encoder, {}, {}, pool);
      decoder_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - decoder_start).count();
      output_count = encoder.size() + decoders.erb.size() + decoders.df.size();
      total_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      durations.push_back(std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count());
      pool.reset();
    }
    const double average_ms = total_ms / iterations;
    std::sort(durations.begin(), durations.end());
    const double p50_ms = durations[durations.size() / 2];
    const double p95_ms = durations[std::min<std::size_t>(durations.size() - 1,
        static_cast<std::size_t>(durations.size() * 0.95))];
    const double max_ms = durations.back();
    std::cout << "backend=" << backend->name() << " iterations=" << iterations
              << " average_ms=" << average_ms << " outputs=" << output_count
              << " encoder_ms=" << encoder_ms / iterations
              << " decoder_ms=" << decoder_ms / iterations
              << " p50_ms=" << p50_ms << " p95_ms=" << p95_ms
              << " max_ms=" << max_ms
              << " realtime_factor_320ms=" << 320.0 / average_ms
              << " per_10ms_ms=" << average_ms / 32.0 << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << "\n";
    return 1;
  }
}
