#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

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
    runtime::HushStreamSession session(model, *backend, {});
    runtime::MemoryPool pool;
    float maximum = 0.0f;
    for (int frame = 0; frame < 100; ++frame) {
      for (int time = 0; time < 32; ++time) {
        std::vector<float> samples(160);
        for (std::size_t index = 0; index < samples.size(); ++index) {
          samples[index] = 0.03f * std::sin(static_cast<float>(index + time * 160 + frame) * 0.11f);
        }
        const auto features = frontend.process(runtime::AudioFrame::mono_16k(samples));
        std::copy_n(features.erb.data<float>(), 32, erb_values.data() + time * 32);
        for (std::size_t channel = 0; channel < 2; ++channel) {
          std::copy_n(features.df.data<float>() + channel * 64, 64,
                      df_values.data() + (channel * 32 + time) * 64);
        }
      }
      const auto outputs = session.process_encoder(
          {{"feat_erb", runtime::Tensor::from_vector({1, 1, 32, 32}, erb_values)},
           {"feat_spec", runtime::Tensor::from_vector({1, 2, 32, 64}, df_values)}}, pool);
      float frame_maximum = 0.0f;
      std::string frame_maximum_name;
      for (const auto& item : outputs) {
        if (item.first.find("Constant") != std::string::npos ||
            item.first.find("Shape") != std::string::npos ||
            item.first.find("Gather") != std::string::npos ||
            item.first.find("Concat") != std::string::npos) continue;
        for (std::size_t index = 0; index < item.second.numel(); ++index) {
          const float value = item.second.data<float>()[index];
          if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite output frame=" + std::to_string(frame) +
                                     " name=" + item.first + " index=" + std::to_string(index));
          }
          maximum = std::max(maximum, std::fabs(value));
          if (std::fabs(value) > frame_maximum) {
            frame_maximum = std::fabs(value);
            frame_maximum_name = item.first;
          }
        }
      }
      if (frame < 3 || frame % 25 == 0) {
        std::cout << "frame=" << frame << " max_abs=" << frame_maximum
                  << " name=" << frame_maximum_name << "\n";
      }
      pool.reset();
    }
    if (session.frame_index() != 100) throw std::runtime_error("frame counter mismatch");
    session.reset();
    if (session.frame_index() != 0) throw std::runtime_error("reset counter mismatch");
    std::cout << "backend=" << backend->name() << " frames=100 max_abs=" << maximum << " stable=true\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "stability failed: " << error.what() << "\n";
    return 1;
  }
}
