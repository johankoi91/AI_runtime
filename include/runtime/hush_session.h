#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/hush_model.h"
#include "runtime/onnx_executor.h"
#include "runtime/audio.h"

namespace runtime {

struct StateBinding {
  std::string input_name;
  std::string output_name;
};

class HushStreamSession {
 public:
  HushStreamSession(
      const HushModel& model,
      const Backend& backend,
      std::vector<StateBinding> encoder_state = {});

  std::unordered_map<std::string, Tensor> process_encoder(
      const std::unordered_map<std::string, Tensor>& inputs,
      MemoryPool& pool);
  std::unordered_map<std::string, Tensor> process_frame(
      const AudioFrame& frame,
      const std::unordered_map<std::string, Tensor>& feature_inputs,
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
  std::size_t frame_index_ = 0;
};

}  // namespace runtime
