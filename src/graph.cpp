#include "runtime/graph.h"

#include <stdexcept>
#include <utility>

namespace runtime {

void Graph::set_input(std::string name, Tensor tensor) {
  tensors_[std::move(name)] = std::move(tensor);
}

void Graph::add_node(Node node) {
  if (node.name.empty()) {
    throw std::runtime_error("graph add_node: node name cannot be empty");
  }
  if (node.inputs.empty()) {
    throw std::runtime_error("graph add_node: node must have inputs");
  }
  nodes_.push_back(std::move(node));
}

bool Graph::has_tensor(const std::string& name) const {
  return tensors_.contains(name);
}

const Tensor& Graph::tensor(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("graph tensor: tensor not found: " + name);
  }
  return it->second;
}

void Graph::store_tensor(const std::string& name, Tensor tensor) {
  tensors_[name] = std::move(tensor);
}

}  // namespace runtime
