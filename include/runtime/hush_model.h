#pragma once

#include <string>
#include <vector>

#include "runtime/hush_bundle.h"
#include "runtime/onnx_weights.h"

namespace runtime {

struct HushConfig {
  std::size_t sample_rate = 16'000;
  std::size_t fft_size = 320;
  std::size_t hop_size = 160;
  std::size_t erb_bands = 32;
  std::size_t min_nb_erb_freqs = 2;
  std::size_t df_bins = 64;
  std::size_t df_order = 5;
  float norm_tau = 1.0f;

  static HushConfig from_ini(const std::string& ini);
  void validate() const;
};

struct HushModel {
  std::string config_ini;
  HushConfig config;
  OnnxGraph encoder;
  OnnxGraph erb_decoder;
  OnnxGraph df_decoder;

  static HushModel load(const std::string& path);
  static HushModel load_for_inspection(const std::string& path);
};

struct HushGraphReport {
  std::size_t node_count = 0;
  std::vector<std::string> op_types;
  std::vector<std::string> unsupported_ops;
  bool has_inputs = false;
  bool has_outputs = false;
};

HushGraphReport inspect_hush_graph(const OnnxGraph& graph);
void validate_hush_model(const HushModel& model);

}  // namespace runtime
