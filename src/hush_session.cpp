#include "runtime/hush_session.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "runtime/kernels.h"

namespace runtime {

namespace {

Tensor owned_copy(const Tensor& input) {
  Tensor output = Tensor::empty(input.shape(), input.dtype());
  std::copy_n(input.data<float>(), input.numel(), output.data<float>());
  return output;
}

Tensor apply_deep_filter(const Tensor& spectrum, const Tensor& coefs, std::size_t df_bins,
                         std::size_t df_order, std::vector<float>& history, MemoryPool& pool) {
  if (spectrum.rank() != 3 || spectrum.shape()[2] != 2 || coefs.rank() != 4 ||
      coefs.shape()[0] != 1 || coefs.shape()[2] != df_bins || coefs.shape()[3] != df_order * 2) {
    return Tensor();
  }
  if (history.size() != (df_order - 1) * df_bins * 2) history.assign((df_order - 1) * df_bins * 2, 0.0f);
  Tensor output = pool.allocate_tensor(spectrum.shape());
  std::copy_n(spectrum.data<float>(), spectrum.numel(), output.data<float>());
  const std::size_t frames = spectrum.shape()[0], bins = spectrum.shape()[1];
  const std::size_t coefficient_frames = coefs.shape()[1];
  for (std::size_t frame = 0; frame < frames; ++frame) {
    const std::size_t coefficient_frame = std::min(frame, coefficient_frames - 1);
    for (std::size_t bin = 0; bin < df_bins; ++bin) {
      float real = 0.0f, imag = 0.0f;
      for (std::size_t tap = 0; tap < df_order; ++tap) {
        const bool current = tap + 1 == df_order;
        const float source_real = current ? spectrum.data<float>()[(frame * bins + bin) * 2]
            : history[(tap * df_bins + bin) * 2];
        const float source_imag = current ? spectrum.data<float>()[(frame * bins + bin) * 2 + 1]
            : history[(tap * df_bins + bin) * 2 + 1];
        const std::size_t coefficient_tap = tap;
        const std::size_t offset = ((coefficient_frame * df_bins + bin) * df_order + coefficient_tap) * 2;
        const float coefficient_real = coefs.data<float>()[offset];
        const float coefficient_imag = coefs.data<float>()[offset + 1];
        real += source_real * coefficient_real - source_imag * coefficient_imag;
        imag += source_imag * coefficient_real + source_real * coefficient_imag;
      }
      output.data<float>()[(frame * bins + bin) * 2] = real;
      output.data<float>()[(frame * bins + bin) * 2 + 1] = imag;
    }
    if (df_order > 1) std::copy(history.begin() + df_bins * 2, history.end(), history.begin());
    if (df_order > 1) std::copy_n(spectrum.data<float>() + frame * bins * 2, df_bins * 2,
                                  history.data() + (df_order - 2) * df_bins * 2);
  }
  return output;
}

std::vector<float> synthesize_hush(const Tensor& spectrum, std::size_t fft_size, std::size_t hop_size,
                                   std::vector<float>& history) {
  constexpr float kPi = 3.14159265358979323846f;
  if (history.size() != fft_size - hop_size) history.assign(fft_size - hop_size, 0.0f);
  const std::size_t bins = spectrum.shape()[1];
  std::vector<float> output;
  output.reserve(spectrum.shape()[0] * hop_size);
  for (std::size_t frame = 0; frame < spectrum.shape()[0]; ++frame) {
    std::vector<float> time(fft_size, 0.0f);
    for (std::size_t sample = 0; sample < fft_size; ++sample) {
      float inverse = 0.0f;
      for (std::size_t bin = 0; bin < fft_size; ++bin) {
        float real = 0.0f, imag = 0.0f;
        if (bin < bins) {
          real = spectrum.data<float>()[(frame * bins + bin) * 2];
          imag = spectrum.data<float>()[(frame * bins + bin) * 2 + 1];
        } else {
          const std::size_t mirror = fft_size - bin;
          if (mirror == 0 || mirror >= bins - 1) continue;
          real = spectrum.data<float>()[(frame * bins + mirror) * 2];
          imag = -spectrum.data<float>()[(frame * bins + mirror) * 2 + 1];
        }
        const float angle = 2.0f * kPi * bin * sample / fft_size;
        inverse += real * std::cos(angle) - imag * std::sin(angle);
      }
      const float phase = 0.5f * kPi * (sample + 0.5f) / (fft_size / 2.0f);
      const float sine = std::sin(phase);
      time[sample] = inverse * std::sin(0.5f * kPi * sine * sine);
    }
    for (std::size_t sample = 0; sample < hop_size; ++sample) output.push_back(time[sample] + history[sample]);
    std::copy(time.begin() + hop_size, time.end(), history.begin());
  }
  return output;
}

}  // namespace

HushStreamSession::HushStreamSession(
    const HushModel& model,
    const Backend& backend,
    std::vector<StateBinding> encoder_state)
    : model_(model), executor_(backend), encoder_state_bindings_(std::move(encoder_state)),
      frontend_(model.config) {
  model_.config.validate();
  executor_.prepare(model_.encoder);
  executor_.prepare(model_.erb_decoder);
  executor_.prepare(model_.df_decoder);
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
    MemoryPool& pool,
    OnnxStreamState* stream_state) {
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

  auto outputs = executor_.run(model_.encoder, frame_inputs, pool, stream_state);
  for (const StateBinding& binding : encoder_state_bindings_) {
    const auto output = outputs.find(binding.output_name);
    if (output == outputs.end()) {
      throw std::runtime_error(
          "Hush encoder state output is missing: " + binding.output_name);
    }
    state_[binding.input_name] = owned_copy(output->second);
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
    MemoryPool& pool,
    OnnxStreamState* erb_state,
    OnnxStreamState* df_state) const {
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
  outputs.erb = executor_.run(model_.erb_decoder, bind_encoder_outputs(erb_inputs), pool, erb_state);
  outputs.df = executor_.run(model_.df_decoder, bind_encoder_outputs(df_inputs), pool, df_state);
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
  for (const OnnxNode& node : model_.encoder.nodes) {
    if (node.op_type == "Conv") {
      return process_audio_batch({frame}, erb_input_name, df_input_name, mask_output_name, backend, pool).front();
    }
  }
  const HushFeatures features = frontend_.process(frame);
  Tensor erb = features.erb;
  Tensor df = features.df;
  if (erb.rank() == 2) erb = cpu::reshape(erb, {1, 1, erb.shape()[0], erb.shape()[1]});
  if (df.rank() == 3) df = cpu::reshape(df, {1, df.shape()[0], df.shape()[1], df.shape()[2]});
  const auto encoder_outputs = process_encoder(
      {{erb_input_name, erb}, {df_input_name, df}}, pool);
  const auto lsnr_iterator = encoder_outputs.find("lsnr");
  const float lsnr = lsnr_iterator != encoder_outputs.end() && lsnr_iterator->second.numel() > 0
      ? lsnr_iterator->second.data<float>()[0] : 0.0f;
  if (lsnr > 35.0f) {
    const auto audio = synthesize_hush(features.spectrum, model_.config.fft_size,
                                       model_.config.hop_size, synthesis_history_);
    return AudioFrame(AudioFormat{model_.config.sample_rate, 1}, audio);
  }
  const auto decoder_outputs = process_decoders(encoder_outputs, {}, {}, pool);
  const auto mask_iterator = decoder_outputs.erb.find(mask_output_name);
  if (mask_iterator == decoder_outputs.erb.end()) {
    throw std::runtime_error("Hush ERB decoder mask output is missing: " + mask_output_name);
  }
  Tensor mask = mask_iterator->second;
  if (lsnr < -15.0f) mask.fill(0.0f);
  if (mask.rank() == 4) {
    const std::size_t bands = mask.shape()[3];
    const std::size_t time = mask.shape()[2] - 1;
    std::vector<float> values(bands);
    std::copy_n(mask.data<float>() + time * bands, bands, values.data());
    mask = Tensor::from_vector({1, bands}, values);
  }
  const Tensor masked = cpu::apply_erb_mask(features.spectrum, mask, frontend_.erb_projection(), &pool);
  Tensor enhanced_spectrum = masked;
  const auto coefficient_iterator = decoder_outputs.df.find("coefs");
  if (coefficient_iterator != decoder_outputs.df.end()) {
    Tensor coefficients = coefficient_iterator->second;
    if (coefficients.rank() == 4 && coefficients.shape()[1] > 1) {
      const std::size_t time = coefficients.shape()[1] - 1;
      const std::size_t size = coefficients.shape()[2] * coefficients.shape()[3];
      std::vector<float> last(size);
      std::copy_n(coefficients.data<float>() + time * size, size, last.data());
      coefficients = Tensor::from_vector({1, 1, coefficients.shape()[2], coefficients.shape()[3]}, last);
    }
    const Tensor deep_filtered = apply_deep_filter(features.spectrum, coefficients,
                                                    model_.config.df_bins, model_.config.df_order,
                                                    df_history_, pool);
    if (deep_filtered.numel() != 0) {
      enhanced_spectrum = deep_filtered;
      const std::size_t low_bins = std::min(model_.config.df_bins, masked.shape()[1]);
      for (std::size_t index = low_bins; index < masked.shape()[1]; ++index) {
        std::copy_n(masked.data<float>() + index * 2, 2,
                    enhanced_spectrum.data<float>() + index * 2);
      }
    }
  }
  const std::vector<float> audio = synthesize_hush(enhanced_spectrum, model_.config.fft_size,
                                                   model_.config.hop_size, synthesis_history_);
  return AudioFrame(AudioFormat{model_.config.sample_rate, 1},
                    audio);
}

std::vector<AudioFrame> HushStreamSession::process_audio_batch(
    const std::vector<AudioFrame>& frames,
    const std::string& erb_input_name,
    const std::string& df_input_name,
    const std::string& mask_output_name,
    const Backend& backend,
    MemoryPool& pool) {
  (void)backend;
  if (frames.empty() || frames.size() > 32) throw std::invalid_argument("Hush batch must contain 1..32 frames");
  const std::size_t time_steps = frames.size();
  const std::size_t erb_bands = model_.config.erb_bands;
  const std::size_t df_bins = model_.config.df_bins;
  std::vector<float> erb_values(time_steps * erb_bands, 0.0f);
  std::vector<float> df_values(2 * time_steps * df_bins, 0.0f);
  std::vector<Tensor> spectra;
  spectra.reserve(time_steps);
  for (std::size_t time = 0; time < frames.size(); ++time) {
    validate_frame(frames[time]);
    const HushFeatures features = frontend_.process(frames[time]);
    spectra.push_back(features.spectrum);
    std::copy_n(features.erb.data<float>(), erb_bands, erb_values.data() + time * erb_bands);
    for (std::size_t bin = 0; bin < df_bins; ++bin) {
      df_values[(time * df_bins) + bin] = features.df.data<float>()[bin];
      df_values[time * df_bins + time_steps * df_bins + bin] = features.df.data<float>()[df_bins + bin];
    }
  }
  Tensor erb = Tensor::from_vector({1, 1, time_steps, erb_bands}, erb_values);
  Tensor df = Tensor::from_vector({1, 2, time_steps, df_bins}, df_values);
  const auto encoder_outputs = process_encoder({{erb_input_name, erb}, {df_input_name, df}}, pool, &encoder_stream_state_);
  const auto decoder_outputs = process_decoders(encoder_outputs, {}, {}, pool, &erb_stream_state_, &df_stream_state_);
  auto mask_iterator = decoder_outputs.erb.find(mask_output_name);
  if (mask_iterator == decoder_outputs.erb.end()) throw std::runtime_error("Hush ERB decoder mask output is missing: " + mask_output_name);
  const Tensor& mask_sequence = mask_iterator->second;
  const auto coefficient_iterator = decoder_outputs.df.find("coefs");
  std::vector<AudioFrame> outputs;
  outputs.reserve(frames.size());
  for (std::size_t time = 0; time < frames.size(); ++time) {
    const std::size_t bands = mask_sequence.shape().back();
    std::vector<float> mask_values(bands);
    const std::size_t mask_time = mask_sequence.rank() == 4 ? std::min(time, mask_sequence.shape()[2] - 1) : 0;
    std::copy_n(mask_sequence.data<float>() + mask_time * bands, bands, mask_values.data());
    const Tensor mask = Tensor::from_vector({1, bands}, mask_values);
    const Tensor masked = cpu::apply_erb_mask(spectra[time], mask, frontend_.erb_projection(), &pool);
    Tensor enhanced_spectrum = masked;
    if (coefficient_iterator != decoder_outputs.df.end()) {
      const Tensor& sequence = coefficient_iterator->second;
      const std::size_t coefficient_size = sequence.shape()[2] * sequence.shape()[3];
      std::vector<float> coefficient_values(coefficient_size);
      std::copy_n(sequence.data<float>() + time * coefficient_size, coefficient_size, coefficient_values.data());
      const Tensor coefficients = Tensor::from_vector({1, 1, sequence.shape()[2], sequence.shape()[3]}, coefficient_values);
      const Tensor deep_filtered = apply_deep_filter(spectra[time], coefficients, model_.config.df_bins,
                                                      model_.config.df_order, df_history_, pool);
      if (deep_filtered.numel() != 0) {
        enhanced_spectrum = deep_filtered;
        const std::size_t low_bins = std::min(model_.config.df_bins, masked.shape()[1]);
        for (std::size_t bin = low_bins; bin < masked.shape()[1]; ++bin) {
          std::copy_n(masked.data<float>() + bin * 2, 2, enhanced_spectrum.data<float>() + bin * 2);
        }
      }
    }
    outputs.emplace_back(AudioFormat{model_.config.sample_rate, 1},
                         synthesize_hush(enhanced_spectrum, model_.config.fft_size,
                                         model_.config.hop_size, synthesis_history_));
  }
  return outputs;
}

void HushStreamSession::reset() {
  state_.clear();
  frontend_.reset();
  df_history_.clear();
  synthesis_history_.clear();
  encoder_stream_state_ = {};
  erb_stream_state_ = {};
  df_stream_state_ = {};
  frame_index_ = 0;
}

bool HushStreamSession::has_state(const std::string& input_name) const {
  return state_.find(input_name) != state_.end();
}

}  // namespace runtime
