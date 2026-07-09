#include "runtime/graph.h"

#include <stdexcept>
#include <utility>

namespace runtime {

void Graph::set_input(std::string name, Tensor tensor) {
  // 输入本质上也是“提前放进图里的命名 tensor”。
  tensors_[std::move(name)] = std::move(tensor);
}

void Graph::add_node(Node node) {
  // 这类校验看起来简单，但它们能帮助我们尽早发现图定义错误。
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
  // 读 tensor 时，如果名字不存在，就直接报错。
  // 这样比返回空指针更适合当前学习阶段。
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    throw std::runtime_error("graph tensor: tensor not found: " + name);
  }
  return it->second;
}

void Graph::store_tensor(const std::string& name, Tensor tensor) {
  // 中间结果和最终结果也都统一存回 tensors_。
  tensors_[name] = std::move(tensor);
}

}  // namespace runtime
