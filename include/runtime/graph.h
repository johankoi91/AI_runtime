#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/tensor.h"

namespace runtime {

enum class OpType {
  Add,
  MatMul,
};

struct Node {
  std::string name;
  OpType op;
  std::vector<std::string> inputs;
};

class Graph {
 public:
  void set_input(std::string name, Tensor tensor);
  void add_node(Node node);

  bool has_tensor(const std::string& name) const;
  const Tensor& tensor(const std::string& name) const;
  void store_tensor(const std::string& name, Tensor tensor);

  const std::vector<Node>& nodes() const { return nodes_; }

 private:
  std::vector<Node> nodes_;
  std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace runtime
