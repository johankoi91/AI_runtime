#include "runtime/executor.h"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace runtime {

namespace {

std::string_view op_type_name(OpType op) {
  switch (op) {
    case OpType::Add:
      return "Add";
    case OpType::MatMul:
      return "MatMul";
  }
  return "Unknown";
}

void log_tensor(std::ostream& out, std::string_view label, const std::string& name, const Tensor& tensor) {
  out << "    " << label << ' ' << name << ": " << tensor.debug_string() << '\n';
}

}  // namespace

void Executor::run(Graph& graph, MemoryPool& pool) const {
  // 当前版本里，Graph 的节点顺序就是执行顺序。
  const auto& nodes = graph.nodes();

  for (std::size_t index = 0; index < nodes.size(); ++index) {
    const Node& node = nodes[index];
    const std::size_t used_before = pool.used_bytes();
    if (trace_ != nullptr) {
      std::string ready_detail =
          (node.op == OpType::MatMul ? "matmul inputs=" : "add inputs=");
      for (std::size_t input_index = 0; input_index < node.inputs.size(); ++input_index) {
        if (input_index > 0) {
          ready_detail += ",";
        }
        ready_detail += node.inputs[input_index];
      }
      trace_->emit("executor", "node_ready", ready_detail, used_before);
      trace_->emit("executor", "node_start", node.name, used_before);
    }

    if (logger_ != nullptr) {
      *logger_ << "[executor] node " << (index + 1) << "/" << nodes.size()
               << " -> " << node.name << " = " << op_type_name(node.op) << "(";
      for (std::size_t i = 0; i < node.inputs.size(); ++i) {
        if (i > 0) {
          *logger_ << ", ";
        }
        *logger_ << node.inputs[i];
      }
      *logger_ << ")\n";
      *logger_ << "  pool used before: " << used_before << " bytes\n";
    }

    switch (node.op) {
      case OpType::Add: {
        if (node.inputs.size() != 2) {
          throw std::runtime_error("executor add: expected 2 inputs");
        }

        // 先从图里取输入，再交给 backend 执行，最后把结果写回图中。
        const Tensor& lhs = graph.tensor(node.inputs[0]);
        const Tensor& rhs = graph.tensor(node.inputs[1]);
        if (trace_ != nullptr) {
          trace_->emit("executor", "inputs_ready", "add lhs+rhs", lhs.nbytes() + rhs.nbytes());
        }
        if (logger_ != nullptr) {
          log_tensor(*logger_, "input ", node.inputs[0], lhs);
          log_tensor(*logger_, "input ", node.inputs[1], rhs);
        }
        graph.store_tensor(node.name, backend_.add(lhs, rhs, pool));
        if (logger_ != nullptr) {
          const Tensor& out = graph.tensor(node.name);
          log_tensor(*logger_, "output", node.name, out);
        }
        break;
      }
      case OpType::MatMul: {
        if (node.inputs.size() != 2) {
          throw std::runtime_error("executor matmul: expected 2 inputs");
        }
        const Tensor& lhs = graph.tensor(node.inputs[0]);
        const Tensor& rhs = graph.tensor(node.inputs[1]);
        if (trace_ != nullptr) {
          trace_->emit("executor", "inputs_ready", "matmul lhs×rhs", lhs.nbytes() + rhs.nbytes());
        }
        if (logger_ != nullptr) {
          log_tensor(*logger_, "input ", node.inputs[0], lhs);
          log_tensor(*logger_, "input ", node.inputs[1], rhs);
        }
        graph.store_tensor(node.name, backend_.matmul(lhs, rhs, pool));
        if (logger_ != nullptr) {
          const Tensor& out = graph.tensor(node.name);
          log_tensor(*logger_, "output", node.name, out);
        }
        break;
      }
    }

    if (logger_ != nullptr) {
      const std::size_t used_after = pool.used_bytes();
      *logger_ << "  pool used after : " << used_after << " bytes"
               << " (delta " << (used_after - used_before) << ")\n\n";
    }
    if (trace_ != nullptr) {
      trace_->emit("executor", "node_complete", node.name, pool.used_bytes() - used_before);
    }
  }
}

}  // namespace runtime
