#pragma once

#include <string>
#include <cstdint>
#include <vector>

#include "runtime/weights.h"

namespace runtime {

struct OnnxNode {
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  struct Attribute {
    std::string name;
    std::int64_t integer = 0;
    std::vector<std::int64_t> integers;
    std::string string;
  };
  std::vector<Attribute> attributes;
};

struct OnnxGraph {
  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<OnnxNode> nodes;
  std::vector<WeightTensor> initializers;

  const WeightTensor& initializer(const std::string& name) const;
  const OnnxNode::Attribute& attribute(
      const OnnxNode& node,
      const std::string& name) const;
};

class OnnxWeights {
 public:
  static std::vector<WeightTensor> read_model(const std::string& path);
  static OnnxGraph read_graph(const std::string& path);
  static OnnxGraph read_graph_bytes(const std::vector<std::uint8_t>& bytes);
};

}  // namespace runtime
