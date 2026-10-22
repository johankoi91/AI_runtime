#include "runtime/hush_session.h"

#include <algorithm>
#include <stdexcept>

#include "runtime/kernels.h"

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

HushDecoderOutputs HushStreamSession::process_decoders(
    const std::unordered_map<std::string, Tensor>& encoder_outputs,
    const std::unordered_map<std::string, Tensor>& erb_inputs,
    const std::unordered_map<std::string, Tensor>& df_inputs,
    MemoryPool& pool) const {
  auto bind_encoder_outputs = [&](const std::unordered_map<std::string, Tensor>& explicit_inputs) {
    std::unordered_map<std::string, Tensor> inputs = encoder_outputs;
    for (const auto& item : explicit_inputs) inputs[item.first] = item.second;
    for (const std::string& input_name : model_.erb_decoder.inputs) {
      if (inputs.find(input_name) == inputs.end()) {
        const auto output = encoder_outputs.find(input_name);
        if (output != encoder_outputs.end()) inputs.emplace(input_name, output->second);
      }
    }
    return inputs;
  };
  HushDecoderOutputs outputs;
  outputs.erb = executor_.run(model_.erb_decoder, bind_encoder_outputs(erb_inputs), pool);
  outputs.df = executor_.run(model_.df_decoder, bind_encoder_outputs(df_inputs), pool);
  return outputs;
}

std::unordered_map<std::string, Tensor> HushStreamSession::process_audio_frame(
    const AudioFrame& frame,
    const std::string& erb_input_name,
    const std::string& df_input_name,
    MemoryPool& pool) {
  validate_frame(frame);
  const HushFeatures features = frontend_.process(frame);
  Tensor erb = features.erb;
  Tensor df = features.df;
  bool spatial_encoder = false;
  for (const OnnxNode& node : model_.encoder.nodes) spatial_encoder = spatial_encoder || node.op_type == "Conv";
  if (spatial_encoder && erb.rank() == 2) erb = cpu::reshape(erb, {1, 1, erb.shape()[0], erb.shape()[1]});
  if (spatial_encoder && df.rank() == 3) df = cpu::reshape(df, {1, df.shape()[0], df.shape()[1], df.shape()[2]});
  return process_encoder({{erb_input_name, erb}, {df_input_name, df}}, pool);
}

AudioFrame HushStreamSession::process_audio(
    const AudioFrame& frame,
    const std::string& erb_input_name,
    const std::string& df_input_name,
    const std::string& mask_output_name,
    const Backend& backend,
    MemoryPool& pool) {
  validate_frame(frame);
  const HushFeatures features = frontend_.process(frame);
  Tensor erb = features.erb;
  Tensor df = features.df;
  bool spatial_encoder = false;
  for (const OnnxNode& node : model_.encoder.nodes) spatial_encoder = spatial_encoder || node.op_type == "Conv";
  if (spatial_encoder && erb.rank() == 2) erb = cpu::reshape(erb, {1, 1, erb.shape()[0], erb.shape()[1]});
  if (spatial_encoder && df.rank() == 3) df = cpu::reshape(df, {1, df.shape()[0], df.shape()[1], df.shape()[2]});
  const auto encoder_outputs = process_encoder(
      {{erb_input_name, erb}, {df_input_name, df}}, pool);
  const auto decoder_outputs = process_decoders(encoder_outputs, {}, {}, pool);
  const auto mask_iterator = decoder_outputs.erb.find(mask_output_name);
  if (mask_iterator == decoder_outputs.erb.end()) {
    throw std::runtime_error("Hush ERB decoder mask output is missing: " + mask_output_name);
  }
  Tensor mask = mask_iterator->second;
  if (mask.rank() == 4) {
    const std::size_t bands = mask.shape()[3];
    const std::size_t time = mask.shape()[2] - 1;
    std::vector<float> values(bands);
    std::copy_n(mask.data<float>() + time * bands, bands, values.data());
    mask = Tensor::from_vector({1, bands}, values);
  }
  const Tensor projection = Tensor::from_vector(
      {features.spectrum.shape()[1], mask.shape()[1]},
      std::vector<float>(features.spectrum.shape()[1] * mask.shape()[1], 1.0f));
  const Tensor masked = cpu::apply_erb_mask(
      features.spectrum, mask, projection, &pool);
  const Tensor audio = backend.istft(
      masked, model_.config.fft_size, model_.config.hop_size, frame.sample_count(), pool);
  return AudioFrame(AudioFormat{model_.config.sample_rate, 1},
                    std::vector<float>(audio.data<float>(), audio.data<float>() + audio.numel()));
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
