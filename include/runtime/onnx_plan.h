#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "runtime/onnx_weights.h"

namespace runtime {

struct PlannedNode {
  std::size_t graph_index = 0;
  std::vector<std::size_t> dependencies;
  std::size_t output_users = 0;
};

class OnnxExecutionPlan {
 public:
  static OnnxExecutionPlan compile(const OnnxGraph& graph);

  const std::vector<PlannedNode>& nodes() const { return nodes_; }
  std::size_t output_users(const std::string& value_name) const;

 private:
  std::vector<PlannedNode> nodes_;
  std::vector<std::pair<std::string, std::size_t>> value_users_;
};

}  // namespace runtime
