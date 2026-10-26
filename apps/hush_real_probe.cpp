#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>

#include "runtime/cpu_backend.h"
#include "runtime/hush_frontend.h"
#include "runtime/hush_model.h"
#include "runtime/hush_session.h"
#include "runtime/metal_backend.h"
#if defined(__APPLE__)
#include "runtime/influxdb_sink.h"
#endif

int main() {
  const char* path = std::getenv("HUSH_MODEL_BUNDLE");
  if (path == nullptr || *path == '\0') {
    std::cerr << "HUSH_MODEL_BUNDLE is required\n";
    return 2;
  }
  try {
    const runtime::HushModel model = runtime::HushModel::load(path);
    for (const auto& node : model.erb_decoder.nodes) {
      if (node.name.find("df_convp") != std::string::npos) {
        std::cerr << "TARGET inputs=";
        for (const auto& input : node.inputs) std::cerr << '[' << input << ']';
        std::cerr << "\n";
      }
    }
    for (const auto& node : model.erb_decoder.nodes) {
      if (node.name.find("df_convp") != std::string::npos) {
        std::cerr << "decoder_node " << node.name << " inputs=";
        for (const auto& input : node.inputs) std::cerr << input << ',';
        std::cerr << "\n";
      }
    }
    for (const auto& initializer : model.encoder.initializers) {
      if (initializer.name.find("df_fc_emb.0.weight") != std::string::npos) {
        std::cout << "weight " << initializer.name << " shape=";
        for (auto dimension : initializer.shape) std::cout << dimension << ',';
        std::cout << "\n";
      }
    }
    for (const auto& initializer : model.erb_decoder.initializers) {
      if (initializer.name.find("onnx::Conv_314") != std::string::npos) {
        std::cout << "erb_weight " << initializer.name << " shape=";
        for (auto dimension : initializer.shape) std::cout << dimension << ',';
        std::cout << "\n";
      }
      if (initializer.name.find("Conv") != std::string::npos) {
        std::cout << "erb_init " << initializer.name << " shape=";
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
        for (const auto& attribute : node.attributes) {
          std::cout << " attr_name " << attribute.name << " integer=" << attribute.integer << "\n";
          if (!attribute.integers.empty()) {
            std::cout << " attr " << attribute.name << "=";
            for (auto value : attribute.integers) std::cout << value << ',';
            std::cout << "\n";
          }
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
      if (node.name.find("combine") != std::string::npos || node.name.find("erb_fc") != std::string::npos) {
        std::cout << "branch node " << node.name << " op=" << node.op_type << " inputs=";
        for (const auto& input : node.inputs) std::cout << input << ',';
        std::cout << " outputs=";
        for (const auto& output : node.outputs) std::cout << output << ',';
        std::cout << "\n";
      }
      if (node.name.find("emb_gru/gru") != std::string::npos) {
        std::cout << "gru node " << node.name << " op=" << node.op_type << " inputs=";
        for (const auto& input : node.inputs) std::cout << input << ',';
        std::cout << " outputs=";
        for (const auto& output : node.outputs) std::cout << output << ',';
        std::cout << "\n";
      }
    }
    for (const auto& node : model.erb_decoder.nodes) {
      if (node.name.find("df_convp") != std::string::npos) {
        std::cout << "decoder node " << node.name << " inputs=";
        for (const auto& input : node.inputs) std::cout << input << ',';
        std::cout << "\n";
      }
    }
    for (const auto& node : model.erb_decoder.nodes) {
      if (node.name.find("df_convp") != std::string::npos) {
        std::cout << "decoder node " << node.name << " inputs=";
        for (const auto& input : node.inputs) std::cout << input << ',';
        std::cout << "\n";
      }
    }
    std::unique_ptr<runtime::Backend> backend;
    if (const char* backend_name = std::getenv("HUSH_BACKEND"); backend_name != nullptr &&
        std::string(backend_name) == "metal") {
      backend = std::make_unique<runtime::MetalBackend>();
    } else {
      backend = std::make_unique<runtime::CPUBackend>();
    }
#if defined(__APPLE__)
    runtime::InfluxDbSink trace;
#endif
    runtime::HushStreamSession session(model, *backend, {});
#if defined(__APPLE__)
    if (trace.enabled()) {
      trace.emit("hush.app", "run_start", "source=real_probe");
      session.set_trace_sink(&trace);
    }
#endif
    runtime::MemoryPool pool;
    runtime::HushFrontend frontend(model.config);
    std::vector<runtime::Tensor> erb_frames;
    std::vector<runtime::Tensor> df_frames;
    for (int index = 0; index < 32; ++index) {
      std::vector<float> samples(160);
      for (std::size_t sample = 0; sample < samples.size(); ++sample) {
        samples[sample] = 0.05f * std::sin(static_cast<float>(sample + index * 160) * 0.07f);
      }
      const runtime::AudioFrame frame = runtime::AudioFrame::mono_16k(samples);
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
    const auto decoder_outputs = session.process_decoders(outputs, {}, {}, pool);
    std::cout << "erb_decoder_outputs=" << decoder_outputs.erb.size()
              << " df_decoder_outputs=" << decoder_outputs.df.size() << "\n";
    for (const auto& output : decoder_outputs.erb) {
      std::cout << "erb:" << output.first << "=" << output.second.debug_string() << "\n";
    }
    for (const auto& output : decoder_outputs.df) {
      std::cout << "df:" << output.first << "=" << output.second.debug_string() << "\n";
    }
    runtime::MemoryPool audio_pool;
    std::vector<float> audio_samples(160);
    for (std::size_t sample = 0; sample < audio_samples.size(); ++sample) {
      audio_samples[sample] = 0.05f * std::sin(static_cast<float>(sample) * 0.07f);
    }
    const runtime::AudioFrame enhanced = session.process_audio(
        runtime::AudioFrame::mono_16k(audio_samples),
        "feat_erb", "feat_spec", "m", *backend, audio_pool);
    std::cout << "enhanced_samples=" << enhanced.sample_count()
              << " first=" << enhanced.samples().front();
    double checksum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (float value : enhanced.samples()) {
      checksum += value;
      sum_abs += std::fabs(value);
      sum_sq += static_cast<double>(value) * value;
      max_abs = std::max(max_abs, std::fabs(value));
    }
    std::cout << " checksum=" << checksum << " sum_abs=" << sum_abs
              << " sum_sq=" << sum_sq << " max_abs=" << max_abs << "\n";
#if defined(__APPLE__)
    if (trace.enabled()) {
      trace.emit("hush.app", "run_complete", "source=real_probe", enhanced.sample_count() * sizeof(float));
      const bool uploaded = trace.flush();
      std::cout << "trace_run_id=" << trace.run_id() << " uploaded=" << uploaded
                << " dropped=" << trace.status().dropped_events << "\n";
      if (!uploaded) return 1;
    }
#endif
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "real Hush probe failed: " << error.what() << "\n";
    return 1;
  }
}
