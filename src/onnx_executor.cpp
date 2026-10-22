#include "runtime/onnx_executor.h"

#include <stdexcept>

namespace runtime {

namespace {

const Tensor& lookup(
    const std::unordered_map<std::string, Tensor>& tensors,
    const std::string& name) {
  const auto iterator = tensors.find(name);
  if (iterator == tensors.end()) throw std::runtime_error("ONNX tensor not found: " + name);
  return iterator->second;
}

}  // namespace

std::unordered_map<std::string, Tensor> OnnxExecutor::run(
    const OnnxGraph& graph,
    const std::unordered_map<std::string, Tensor>& inputs,
    MemoryPool& pool) const {
  std::unordered_map<std::string, Tensor> tensors = inputs;
  for (const WeightTensor& initializer : graph.initializers) {
    tensors.emplace(
        initializer.name,
        Tensor::from_vector(initializer.shape, initializer.values));
  }

  for (const OnnxNode& node : graph.nodes) {
    if (node.inputs.size() != 2 || node.outputs.size() != 1) {
      throw std::runtime_error("ONNX node must have two inputs and one output: " + node.name);
    }
    const Tensor& lhs = lookup(tensors, node.inputs[0]);
    const Tensor& rhs = lookup(tensors, node.inputs[1]);
    Tensor output;
    if (node.op_type == "Add") {
      output = backend_.add(lhs, rhs, pool);
    } else if (node.op_type == "MatMul") {
      output = backend_.matmul(lhs, rhs, pool);
    } else {
      throw std::runtime_error("unsupported ONNX op in stage 14: " + node.op_type);
    }
    tensors[node.outputs[0]] = std::move(output);
  }
  return tensors;
}

}  // namespace runtime
