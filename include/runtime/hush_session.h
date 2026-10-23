#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/hush_model.h"
#include "runtime/onnx_executor.h"
#include "runtime/audio.h"
#include "runtime/hush_frontend.h"

namespace runtime {

struct StateBinding {
  std::string input_name;
  std::string output_name;
};

struct HushDecoderOutputs {
  std::unordered_map<std::string, Tensor> erb;
  std::unordered_map<std::string, Tensor> df;
};

class HushStreamSession {
 public:
  HushStreamSession(
      const HushModel& model,
      const Backend& backend,
      std::vector<StateBinding> encoder_state = {});

  std::unordered_map<std::string, Tensor> process_encoder(
      const std::unordered_map<std::string, Tensor>& inputs,
      MemoryPool& pool,
      OnnxStreamState* stream_state = nullptr);
  std::unordered_map<std::string, Tensor> process_frame(
      const AudioFrame& frame,
      const std::unordered_map<std::string, Tensor>& feature_inputs,
      MemoryPool& pool);
  std::unordered_map<std::string, Tensor> process_audio_frame(
      const AudioFrame& frame,
      const std::string& feature_input_name,
      MemoryPool& pool);
  HushDecoderOutputs process_decoders(
      const std::unordered_map<std::string, Tensor>& encoder_outputs,
      const std::unordered_map<std::string, Tensor>& erb_inputs,
      const std::unordered_map<std::string, Tensor>& df_inputs,
      MemoryPool& pool,
      OnnxStreamState* erb_state = nullptr,
      OnnxStreamState* df_state = nullptr) const;
  std::unordered_map<std::string, Tensor> process_audio_frame(
      const AudioFrame& frame,
      const std::string& erb_input_name,
      const std::string& df_input_name,
      MemoryPool& pool);
  AudioFrame process_audio(
      const AudioFrame& frame,
      const std::string& erb_input_name,
      const std::string& df_input_name,
      const std::string& mask_output_name,
      const Backend& backend,
      MemoryPool& pool);
  std::vector<AudioFrame> process_audio_batch(
      const std::vector<AudioFrame>& frames,
      const std::string& erb_input_name,
      const std::string& df_input_name,
      const std::string& mask_output_name,
      const Backend& backend,
      MemoryPool& pool);
  void reset();

  std::size_t frame_index() const { return frame_index_; }
  bool has_state(const std::string& input_name) const;
  void validate_frame(const AudioFrame& frame) const;

 private:
  const HushModel& model_;
  OnnxExecutor executor_;
  std::vector<StateBinding> encoder_state_bindings_;
  std::unordered_map<std::string, Tensor> state_;
  HushFrontend frontend_;
  std::vector<float> df_history_;
  std::vector<float> synthesis_history_;
  OnnxStreamState encoder_stream_state_;
  OnnxStreamState erb_stream_state_;
  OnnxStreamState df_stream_state_;
  std::size_t frame_index_ = 0;
};

}  // namespace runtime
