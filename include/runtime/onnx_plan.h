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
  bool is_graph_output(const std::string& value_name) const;
  std::size_t peak_live_values() const { return peak_live_values_; }

 private:
  std::vector<PlannedNode> nodes_;
  std::vector<std::pair<std::string, std::size_t>> value_users_;
  std::size_t peak_live_values_ = 0;
  std::vector<std::string> graph_outputs_;
};

}  // namespace runtime
