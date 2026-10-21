#pragma once

#include <string>

#include "runtime/hush_bundle.h"
#include "runtime/onnx_weights.h"

namespace runtime {

struct HushModel {
  std::string config_ini;
  OnnxGraph encoder;
  OnnxGraph erb_decoder;
  OnnxGraph df_decoder;

  static HushModel load(const std::string& path);
};

}  // namespace runtime
