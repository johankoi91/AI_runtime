#include "runtime/hush_session.h"

#include <algorithm>
#include <stdexcept>

namespace runtime {

HushStreamSession::HushStreamSession(
    const HushModel& model,
    const Backend& backend,
    std::vector<StateBinding> encoder_state)
    : model_(model), executor_(backend), encoder_state_bindings_(std::move(encoder_state)),
      frontend_(model.config) {
  model_.config.validate();
  if (model_.encoder.nodes.empty()) {
    throw std::invalid_argument("Hush encoder graph must contain at least one node");
  }
  for (const StateBinding& binding : encoder_state_bindings_) {
    bool output_found = false;
    for (const OnnxNode& node : model_.encoder.nodes) {
      for (const std::string& output : node.outputs) {
        output_found = output_found || output == binding.output_name;
      }
    }
    if (!output_found) {
      throw std::invalid_argument(
          "Hush encoder state output is not produced by graph: " + binding.output_name);
    }
  }
}

void HushStreamSession::validate_frame(const AudioFrame& frame) const {
  if (frame.format().sample_rate != model_.config.sample_rate ||
      frame.format().channels != 1 || frame.sample_count() != model_.config.hop_size) {
    throw std::invalid_argument("Hush frame does not match model audio configuration");
  }
}

std::unordered_map<std::string, Tensor> HushStreamSession::process_encoder(
    const std::unordered_map<std::string, Tensor>& inputs,
    MemoryPool& pool) {
  std::unordered_map<std::string, Tensor> frame_inputs = inputs;
  for (const std::string& input_name : model_.encoder.inputs) {
    bool is_initializer = false;
    for (const WeightTensor& initializer : model_.encoder.initializers) {
      is_initializer = is_initializer || initializer.name == input_name;
    }
    const bool is_state = std::any_of(
        encoder_state_bindings_.begin(), encoder_state_bindings_.end(),
        [&](const StateBinding& binding) { return binding.input_name == input_name; });
    if (!is_initializer && !is_state && frame_inputs.find(input_name) == frame_inputs.end()) {
      throw std::runtime_error("Hush encoder input is missing: " + input_name);
    }
  }
  for (const StateBinding& binding : encoder_state_bindings_) {
    if (frame_inputs.find(binding.input_name) == frame_inputs.end()) {
      const auto state = state_.find(binding.input_name);
      if (state == state_.end()) {
        throw std::runtime_error(
            "Hush encoder state is missing on first frame: " + binding.input_name);
      }
      frame_inputs.emplace(binding.input_name, state->second);
    }
  }

  auto outputs = executor_.run(model_.encoder, frame_inputs, pool);
  for (const StateBinding& binding : encoder_state_bindings_) {
    const auto output = outputs.find(binding.output_name);
    if (output == outputs.end()) {
      throw std::runtime_error(
          "Hush encoder state output is missing: " + binding.output_name);
    }
    state_[binding.input_name] = output->second;
  }
  ++frame_index_;
  return outputs;
}

std::unordered_map<std::string, Tensor> HushStreamSession::process_frame(
    const AudioFrame& frame,
    const std::unordered_map<std::string, Tensor>& feature_inputs,
    MemoryPool& pool) {
  validate_frame(frame);
  return process_encoder(feature_inputs, pool);
}

std::unordered_map<std::string, Tensor> HushStreamSession::process_audio_frame(
    const AudioFrame& frame,
    const std::string& feature_input_name,
    MemoryPool& pool) {
  validate_frame(frame);
  const HushFeatures features = frontend_.process(frame);
  return process_encoder({{feature_input_name, features.erb}}, pool);
}

void HushStreamSession::reset() {
  state_.clear();
  frontend_.reset();
  frame_index_ = 0;
}

bool HushStreamSession::has_state(const std::string& input_name) const {
  return state_.find(input_name) != state_.end();
}

}  // namespace runtime
