#include "runtime/executor.h"

#include <stdexcept>

namespace runtime {

void Executor::run(Graph& graph, MemoryPool& pool) const {
  for (const Node& node : graph.nodes()) {
    switch (node.op) {
      case OpType::Add: {
        if (node.inputs.size() != 2) {
          throw std::runtime_error("executor add: expected 2 inputs");
        }
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
