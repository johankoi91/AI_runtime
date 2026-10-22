#include <cstdlib>
#include <iostream>
#include <vector>

#include "runtime/cpu_backend.h"
#include "runtime/hush_frontend.h"
#include "runtime/hush_model.h"
#include "runtime/hush_session.h"

int main() {
  const char* path = std::getenv("HUSH_MODEL_BUNDLE");
  if (path == nullptr || *path == '\0') {
    std::cerr << "HUSH_MODEL_BUNDLE is required\n";
    return 2;
  }
  try {
    const runtime::HushModel model = runtime::HushModel::load(path);
    for (const auto& initializer : model.encoder.initializers) {
      if (initializer.name.find("df_fc_emb.0.weight") != std::string::npos) {
        std::cout << "weight " << initializer.name << " shape=";
        for (auto dimension : initializer.shape) std::cout << dimension << ',';
        std::cout << "\n";
      }
    }
    for (const auto& node : model.encoder.nodes) {
      if (node.name.find("df_conv") != std::string::npos && node.op_type == "Conv") {
        std::cout << "conv " << node.name << " inputs=" << node.inputs[1] << "\n";
        try {
          const auto& weight = model.encoder.initializer(node.inputs[1]);
          std::cout << " weight_shape=";
          for (auto dimension : weight.shape) std::cout << dimension << ',';
          std::cout << "\n";
        } catch (...) {
        }
      }
      bool mentions = node.name.find("df_fc_emb") != std::string::npos;
      for (const auto& output : node.outputs) mentions = mentions || output == "/Reshape_output_0";
      if (mentions) {
        std::cout << "node " << node.name << " op=" << node.op_type << " inputs=";
        for (const auto& input : node.inputs) std::cout << input << ',';
        std::cout << " outputs=";
        for (const auto& output : node.outputs) std::cout << output << ',';
        std::cout << "\n";
        for (const auto& attribute : node.attributes) {
          if (!attribute.string.empty()) std::cout << " attr " << attribute.name << "=" << attribute.string << "\n";
        }
      }
    }
    runtime::CPUBackend backend;
    runtime::HushStreamSession session(model, backend, {});
    runtime::MemoryPool pool;
    runtime::HushFrontend frontend(model.config);
    std::vector<runtime::Tensor> erb_frames;
    std::vector<runtime::Tensor> df_frames;
    for (int index = 0; index < 32; ++index) {
      const runtime::AudioFrame frame = runtime::AudioFrame::mono_16k(std::vector<float>(160, 0.0f));
      const runtime::HushFeatures features = frontend.process(frame);
      erb_frames.push_back(features.erb);
      df_frames.push_back(features.df);
    }
    std::vector<float> erb_values(32 * 32);
    std::vector<float> df_values(2 * 32 * 64);
    for (std::size_t time = 0; time < 32; ++time) {
      std::copy_n(erb_frames[time].data<float>(), 32, erb_values.data() + time * 32);
      for (std::size_t channel = 0; channel < 2; ++channel) {
        std::copy_n(df_frames[time].data<float>() + channel * 64, 64,
                    df_values.data() + (channel * 32 + time) * 64);
      }
    }
    const auto outputs = session.process_encoder(
        {{"feat_erb", runtime::Tensor::from_vector({1, 1, 32, 32}, erb_values)},
         {"feat_spec", runtime::Tensor::from_vector({1, 2, 32, 64}, df_values)}}, pool);
    std::cout << "encoder_outputs=" << outputs.size() << " frame=" << session.frame_index() << "\n";
    for (const auto& output : outputs) {
      std::cout << output.first << "=" << output.second.debug_string() << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "real Hush probe failed: " << error.what() << "\n";
    return 1;
  }
}
