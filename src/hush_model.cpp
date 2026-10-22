#include "runtime/hush_model.h"

#include <stdexcept>
#include <cctype>
#include <sstream>
#include <algorithm>

namespace runtime {

namespace {

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::size_t parse_size(const std::string& key, const std::string& value) {
  try {
    std::size_t consumed = 0;
    const unsigned long parsed = std::stoul(value, &consumed);
    if (consumed != value.size() || parsed == 0) throw std::invalid_argument("invalid");
    return static_cast<std::size_t>(parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid Hush config value for " + key + ": " + value);
  }
}

const std::vector<std::uint8_t>& find_entry(
    const HushModelBundle& bundle,
    const std::string& expected_name) {
  if (bundle.has(expected_name)) return bundle.entry(expected_name);
  for (const BundleEntry& item : bundle.entries()) {
    if (item.name.size() > expected_name.size() &&
        item.name.compare(item.name.size() - expected_name.size(), expected_name.size(), expected_name) == 0) {
      return item.data;
    }
  }
  throw std::runtime_error("Hush model bundle is missing " + expected_name);
}

OnnxGraph read_graph_entry(const HushModelBundle& bundle, const std::string& name) {
  return OnnxWeights::read_graph_bytes(find_entry(bundle, name));
}

}  // namespace

HushGraphReport inspect_hush_graph(const OnnxGraph& graph) {
  static const std::vector<std::string> supported = {
      "Add", "Mul", "MatMul", "Relu", "Sigmoid", "Tanh", "Conv", "GRU",
      "Reshape", "Transpose", "Concat", "Flatten"};
  HushGraphReport report;
  report.node_count = graph.nodes.size();
  report.has_inputs = !graph.inputs.empty();
  report.has_outputs = !graph.outputs.empty();
  for (const OnnxNode& node : graph.nodes) {
    report.op_types.push_back(node.op_type);
    if (std::find(supported.begin(), supported.end(), node.op_type) == supported.end() &&
        std::find(report.unsupported_ops.begin(), report.unsupported_ops.end(), node.op_type) ==
            report.unsupported_ops.end()) {
      report.unsupported_ops.push_back(node.op_type);
    }
  }
  return report;
}

void validate_hush_model(const HushModel& model) {
  model.config.validate();
  const HushGraphReport reports[] = {
      inspect_hush_graph(model.encoder),
      inspect_hush_graph(model.erb_decoder),
      inspect_hush_graph(model.df_decoder),
  };
  for (const HushGraphReport& report : reports) {
    if (report.node_count == 0 || !report.has_inputs || !report.has_outputs ||
        !report.unsupported_ops.empty()) {
      throw std::invalid_argument("Hush model graph violates the runtime graph contract");
    }
  }
}

HushConfig HushConfig::from_ini(const std::string& ini) {
  HushConfig config;
  std::istringstream stream(ini);
  std::string line;
  bool in_df = false;
  while (std::getline(stream, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#' || line[0] == ';') continue;
    if (line.front() == '[' && line.back() == ']') {
      in_df = trim(line.substr(1, line.size() - 2)) == "df";
      continue;
    }
    if (!in_df) continue;
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos) throw std::invalid_argument("invalid Hush config line: " + line);
    const std::string key = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));
    if (key == "sr") config.sample_rate = parse_size(key, value);
    else if (key == "fft_size") config.fft_size = parse_size(key, value);
    else if (key == "hop_size") config.hop_size = parse_size(key, value);
    else if (key == "nb_erb") config.erb_bands = parse_size(key, value);
    else if (key == "nb_df") config.df_bins = parse_size(key, value);
    else if (key == "df_order") config.df_order = parse_size(key, value);
  }
  config.validate();
  return config;
}

void HushConfig::validate() const {
  if (sample_rate == 0 || fft_size == 0 || hop_size == 0 || hop_size > fft_size ||
      erb_bands == 0 || df_bins == 0 || df_order == 0) {
    throw std::invalid_argument("invalid Hush model configuration");
  }
}

HushModel HushModel::load(const std::string& path) {
  const HushModelBundle bundle = HushModelBundle::read(path);
  const auto& config = find_entry(bundle, "config.ini");
  HushModel model;
  model.config_ini.assign(reinterpret_cast<const char*>(config.data()), config.size());
  model.config = HushConfig::from_ini(model.config_ini);
  model.encoder = read_graph_entry(bundle, "enc.onnx");
  model.erb_decoder = read_graph_entry(bundle, "erb_dec.onnx");
  model.df_decoder = read_graph_entry(bundle, "df_dec.onnx");
  validate_hush_model(model);
  return model;
}

}  // namespace runtime
