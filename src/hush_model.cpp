#include "runtime/hush_model.h"

#include <stdexcept>

namespace runtime {

namespace {

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

HushModel HushModel::load(const std::string& path) {
  const HushModelBundle bundle = HushModelBundle::read(path);
  const auto& config = find_entry(bundle, "config.ini");
  HushModel model;
  model.config_ini.assign(reinterpret_cast<const char*>(config.data()), config.size());
  model.encoder = read_graph_entry(bundle, "enc.onnx");
  model.erb_decoder = read_graph_entry(bundle, "erb_dec.onnx");
  model.df_decoder = read_graph_entry(bundle, "df_dec.onnx");
  return model;
}

}  // namespace runtime
