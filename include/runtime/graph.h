#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/tensor.h"

namespace runtime {

// 目前的图只支持两种 op，故意保持最小化。
enum class OpType {
  Add,
  MatMul,
};

// Node 表示计算图中的一个节点。
//
// 例如：
// hidden = matmul(a, b)
//
// 那么：
// - name = "hidden"
// - op = OpType::MatMul
// - inputs = {"a", "b"}
struct Node {
  std::string name;
  OpType op;
  std::vector<std::string> inputs;
};

// Graph 负责存两类东西：
// 1. nodes：执行顺序上的“计算步骤”
// 2. tensors：图中已经存在的命名 tensor
//
// 现在这版图非常简单：
// - 不做拓扑排序
// - 默认你 add_node 的顺序就是执行顺序
// - 输入和中间结果都按字符串名字放到 map 里
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

  // 这里像一个“名字到 tensor 的小仓库”。
  std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace runtime
