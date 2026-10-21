#pragma once

#include <string>
#include <vector>

#include "runtime/weights.h"

namespace runtime {

struct OnnxNode {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

struct OnnxGraph {
  std::string name;
  std::vector<OnnxNode> nodes;
  std::vector<WeightTensor> initializers;
};

class OnnxWeights {
 public:
  static std::vector<WeightTensor> read_model(const std::string& path);
  static OnnxGraph read_graph(const std::string& path);
};

}  // namespace runtime
