#include "runtime/executor.h"

#include <stdexcept>

namespace runtime {

void Executor::run(Graph& graph, MemoryPool& pool) const {
  // 当前版本里，Graph 的节点顺序就是执行顺序。
  for (const Node& node : graph.nodes()) {
    switch (node.op) {
      case OpType::Add: {
        if (node.inputs.size() != 2) {
          throw std::runtime_error("executor add: expected 2 inputs");
        }

        // 先从图里取输入，再交给 backend 执行，最后把结果写回图中。
        const Tensor& lhs = graph.tensor(node.inputs[0]);
        const Tensor& rhs = graph.tensor(node.inputs[1]);
        graph.store_tensor(node.name, backend_.add(lhs, rhs, pool));
        break;
      }
      case OpType::MatMul: {
        if (node.inputs.size() != 2) {
          throw std::runtime_error("executor matmul: expected 2 inputs");
        }
        const Tensor& lhs = graph.tensor(node.inputs[0]);
        const Tensor& rhs = graph.tensor(node.inputs[1]);
        graph.store_tensor(node.name, backend_.matmul(lhs, rhs, pool));
        break;
      }
    }
  }
}

}  // namespace runtime
