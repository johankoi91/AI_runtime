#include "runtime/onnx_executor.h"

#include <stdexcept>

namespace runtime {

namespace {

const Tensor& lookup(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name) {
  const auto iterator = tensors.find(name);
  if (iterator == tensors.end()) throw std::runtime_error("ONNX tensor not found: " + name);
  return iterator->second;
}

std::size_t attribute_size(const OnnxGraph& graph, const OnnxNode& node, const std::string& name,
                           std::size_t fallback) {
  try {
    const auto& attribute = graph.attribute(node, name);
    return static_cast<std::size_t>(attribute.integers.empty() ? attribute.integer : attribute.integers.front());
  } catch (const std::out_of_range&) {
    return fallback;
  }
}

Tensor run_gru_step(const Backend& backend, const OnnxNode& node,
                   const std::unordered_map<std::string, Tensor>& tensors, MemoryPool& pool) {
  if (node.inputs.size() < 5) throw std::runtime_error("stage 15 GRU requires X,W,R,B,initial_h");
  const Tensor& sequence = lookup(tensors, node.inputs[0]);
  const Tensor& input_weights = lookup(tensors, node.inputs[1]);
  const Tensor& hidden_weights = lookup(tensors, node.inputs[2]);
  const Tensor& biases = lookup(tensors, node.inputs[3]);
  const Tensor& initial_hidden = lookup(tensors, node.inputs[4]);
  if (sequence.rank() != 3 || sequence.shape()[0] != 1 || sequence.shape()[1] != 1 ||
      input_weights.rank() != 3 || hidden_weights.rank() != 3 || biases.rank() != 2 ||
      initial_hidden.rank() != 3 || input_weights.shape()[0] != 1 || hidden_weights.shape()[0] != 1 ||
      biases.shape()[0] != 1 || initial_hidden.shape()[0] != 1 || initial_hidden.shape()[1] != 1) {
    throw std::runtime_error("stage 15 GRU only supports one direction and one input step");
  }
  const std::size_t hidden_size = hidden_weights.shape()[1] / 3;
  const std::size_t input_size = input_weights.shape()[2];
  if (input_weights.shape() != Shape{1, 3 * hidden_size, input_size} ||
      hidden_weights.shape() != Shape{1, 3 * hidden_size, hidden_size} ||
      biases.shape() != Shape{1, 6 * hidden_size} ||
      initial_hidden.shape() != Shape{1, 1, hidden_size} || sequence.shape()[2] != input_size) {
    throw std::runtime_error("stage 15 GRU tensor shapes are unsupported");
  }
  std::vector<float> bias_values(3 * hidden_size);
  for (std::size_t index = 0; index < bias_values.size(); ++index) {
    bias_values[index] = biases.data<float>()[index] + biases.data<float>()[3 * hidden_size + index];
  }
  Tensor input = Tensor::from_vector({input_size}, std::vector<float>(
      sequence.data<float>(), sequence.data<float>() + input_size));
  Tensor hidden = Tensor::from_vector({hidden_size}, std::vector<float>(
      initial_hidden.data<float>(), initial_hidden.data<float>() + hidden_size));
  Tensor weights = Tensor::from_vector({3 * hidden_size, input_size}, std::vector<float>(
      input_weights.data<float>(), input_weights.data<float>() + 3 * hidden_size * input_size));
  Tensor recurrent = Tensor::from_vector({3 * hidden_size, hidden_size}, std::vector<float>(
      hidden_weights.data<float>(), hidden_weights.data<float>() + 3 * hidden_size * hidden_size));
  Tensor bias = Tensor::from_vector({3 * hidden_size}, bias_values);
  Tensor step = backend.gru_step(input, hidden, weights, recurrent, bias, pool);
  return Tensor::from_vector({1, 1, hidden_size}, std::vector<float>(
      step.data<float>(), step.data<float>() + hidden_size));
}

}  // namespace

std::unordered_map<std::string, Tensor> OnnxExecutor::run(
    const OnnxGraph& graph, const std::unordered_map<std::string, Tensor>& inputs, MemoryPool& pool) const {
  std::unordered_map<std::string, Tensor> tensors = inputs;
  for (const WeightTensor& initializer : graph.initializers) {
    tensors.emplace(initializer.name, Tensor::from_vector(initializer.shape, initializer.values));
  }
  for (const OnnxNode& node : graph.nodes) {
    if (node.outputs.size() != 1) throw std::runtime_error("ONNX node must have one output: " + node.name);
    Tensor output;
    if (node.op_type == "Add" || node.op_type == "MatMul") {
      if (node.inputs.size() != 2) throw std::runtime_error("binary ONNX node requires two inputs");
      const Tensor& lhs = lookup(tensors, node.inputs[0]);
      const Tensor& rhs = lookup(tensors, node.inputs[1]);
      output = node.op_type == "Add" ? backend_.add(lhs, rhs, pool) : backend_.matmul(lhs, rhs, pool);
    } else if (node.op_type == "Relu" || node.op_type == "Sigmoid" || node.op_type == "Tanh") {
      if (node.inputs.size() != 1) throw std::runtime_error("activation node requires one input");
      const Tensor& input = lookup(tensors, node.inputs[0]);
      output = node.op_type == "Relu" ? backend_.relu(input, pool)
          : node.op_type == "Sigmoid" ? backend_.sigmoid(input, pool) : backend_.tanh(input, pool);
    } else if (node.op_type == "Conv") {
      if (node.inputs.size() < 2 || node.inputs.size() > 3) throw std::runtime_error("Conv input count unsupported");
      const Tensor& input = lookup(tensors, node.inputs[0]);
      const Tensor& weights = lookup(tensors, node.inputs[1]);
      Tensor bias = node.inputs.size() == 3 ? lookup(tensors, node.inputs[2]) : Tensor::zeros({weights.shape()[0]});
      output = backend_.conv2d(input, weights, bias, attribute_size(graph, node, "strides", 1),
                               attribute_size(graph, node, "pads", 0), pool);
    } else if (node.op_type == "GRU") {
      output = run_gru_step(backend_, node, tensors, pool);
    } else {
      throw std::runtime_error("unsupported ONNX op in stage 15: " + node.op_type);
    }
    tensors[node.outputs[0]] = std::move(output);
  }
  return tensors;
}

}  // namespace runtime
