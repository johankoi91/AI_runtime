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
    for (const auto& node : model.encoder.nodes) {
      bool mentions = node.name.find("df_fc_emb") != std::string::npos;
      for (const auto& output : node.outputs) mentions = mentions || output == "/Reshape_output_0";
      if (mentions) {
        std::cout << "node " << node.name << " op=" << node.op_type << " inputs=";
        for (const auto& input : node.inputs) std::cout << input << ',';
        std::cout << " outputs=";
        for (const auto& output : node.outputs) std::cout << output << ',';
        std::cout << "\n";
      }
    }
    runtime::CPUBackend backend;
    runtime::HushStreamSession session(model, backend, {});
    runtime::MemoryPool pool;
    const runtime::AudioFrame frame = runtime::AudioFrame::mono_16k(std::vector<float>(160, 0.0f));
    const auto outputs = session.process_audio_frame(frame, "feat_erb", "feat_spec", pool);
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
