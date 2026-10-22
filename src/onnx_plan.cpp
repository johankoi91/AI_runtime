#include "runtime/onnx_plan.h"

#include <queue>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace runtime {

OnnxExecutionPlan OnnxExecutionPlan::compile(const OnnxGraph& graph) {
  OnnxExecutionPlan plan;
  std::unordered_map<std::string, std::size_t> producer;
  for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
    for (const std::string& output : graph.nodes[index].outputs) {
      if (!producer.emplace(output, index).second) throw std::invalid_argument("duplicate ONNX value producer: " + output);
    }
  }
  std::vector<std::vector<std::size_t>> outgoing(graph.nodes.size());
  std::vector<std::size_t> indegree(graph.nodes.size(), 0);
  plan.nodes_.resize(graph.nodes.size());
  for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
    plan.nodes_[index].graph_index = index;
    for (const std::string& input : graph.nodes[index].inputs) {
      const auto found = producer.find(input);
      if (found == producer.end()) continue;
      plan.nodes_[index].dependencies.push_back(found->second);
      outgoing[found->second].push_back(index);
      ++indegree[index];
    }
  }
  std::queue<std::size_t> ready;
  for (std::size_t index = 0; index < indegree.size(); ++index) if (indegree[index] == 0) ready.push(index);
  std::vector<PlannedNode> ordered;
  while (!ready.empty()) {
    const std::size_t index = ready.front(); ready.pop();
    ordered.push_back(plan.nodes_[index]);
    for (std::size_t dependent : outgoing[index]) if (--indegree[dependent] == 0) ready.push(dependent);
  }
  if (ordered.size() != graph.nodes.size()) throw std::invalid_argument("ONNX graph contains a cycle");
  plan.nodes_ = std::move(ordered);

  for (const auto& node : graph.nodes) {
    for (const std::string& input : node.inputs) {
      auto found = std::find_if(plan.value_users_.begin(), plan.value_users_.end(),
          [&](const auto& item) { return item.first == input; });
      if (found == plan.value_users_.end()) plan.value_users_.push_back({input, 1});
      else ++found->second;
    }
  }
  return plan;
}

std::size_t OnnxExecutionPlan::output_users(const std::string& value_name) const {
  for (const auto& item : value_users_) if (item.first == value_name) return item.second;
  return 0;
}

}  // namespace runtime
