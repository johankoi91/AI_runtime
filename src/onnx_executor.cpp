#include "runtime/onnx_executor.h"

#include <numeric>
#include <stdexcept>

#include "runtime/kernels.h"

namespace runtime {

namespace {

std::size_t product(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{});
}

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

std::vector<std::int64_t> attribute_values(
    const OnnxGraph& graph,
    const OnnxNode& node,
    const std::string& name,
    std::vector<std::int64_t> fallback) {
  try {
    const auto& attribute = graph.attribute(node, name);
    if (!attribute.integers.empty()) return attribute.integers;
    return {attribute.integer};
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
      const auto strides = attribute_values(graph, node, "strides", {1, 1});
      const auto pads = attribute_values(graph, node, "pads", {0, 0, 0, 0});
      const auto dilations = attribute_values(graph, node, "dilations", {1, 1});
      const auto groups = attribute_values(graph, node, "group", {1});
      if (strides.size() != 2 || pads.size() != 4 || dilations.size() != 2 || groups.size() != 1 ||
          strides[0] <= 0 || strides[1] <= 0 || pads[0] < 0 || pads[1] < 0 ||
          pads[2] < 0 || pads[3] < 0 || dilations[0] != 1 || dilations[1] != 1 || groups[0] != 1 ||
          pads[0] != pads[2] || pads[1] != pads[3] || strides[0] != strides[1]) {
        throw std::runtime_error("ONNX Conv attributes are outside the supported 2D symmetric subset");
      }
      output = backend_.conv2d(input, weights, bias, static_cast<std::size_t>(strides[0]),
                               static_cast<std::size_t>(pads[0]), pool);
    } else if (node.op_type == "GRU") {
      output = run_gru_step(backend_, node, tensors, pool);
    } else if (node.op_type == "Reshape") {
      if (node.inputs.size() != 2) throw std::runtime_error("Reshape requires data and shape inputs");
      const Tensor& shape_tensor = lookup(tensors, node.inputs[1]);
      std::vector<std::size_t> shape;
      for (std::size_t index = 0; index < shape_tensor.numel(); ++index) {
        const float value = shape_tensor.data<float>()[index];
        if (value < 0.0f) throw std::runtime_error("Reshape negative dimensions are unsupported");
        shape.push_back(static_cast<std::size_t>(value));
      }
      output = cpu::reshape(lookup(tensors, node.inputs[0]), shape);
    } else if (node.op_type == "Transpose") {
      std::vector<std::size_t> permutation;
      try {
        const auto& attribute = graph.attribute(node, "perm");
        for (std::int64_t value : attribute.integers) permutation.push_back(static_cast<std::size_t>(value));
      } catch (const std::out_of_range&) {
        for (std::size_t index = lookup(tensors, node.inputs[0]).rank(); index-- > 0;) permutation.push_back(index);
      }
      output = cpu::transpose(lookup(tensors, node.inputs[0]), permutation, &pool);
    } else if (node.op_type == "Concat") {
      const auto& attribute = graph.attribute(node, "axis");
      std::vector<const Tensor*> concat_inputs;
      for (const std::string& input_name : node.inputs) concat_inputs.push_back(&lookup(tensors, input_name));
      output = cpu::concat(concat_inputs, static_cast<std::size_t>(attribute.integer), &pool);
    } else if (node.op_type == "Flatten") {
      if (node.inputs.size() != 1) throw std::runtime_error("Flatten requires one input");
      const Tensor& input = lookup(tensors, node.inputs[0]);
      const std::size_t axis = attribute_size(graph, node, "axis", 1);
      if (axis > input.rank()) throw std::runtime_error("Flatten axis is out of range");
      const std::size_t left = product(Shape(input.shape().begin(), input.shape().begin() + axis));
      const std::size_t right = input.numel() / left;
      output = cpu::reshape(input, {left, right});
    } else {
      throw std::runtime_error("unsupported ONNX op in stage 15: " + node.op_type);
    }
    tensors[node.outputs[0]] = std::move(output);
  }
  return tensors;
}

}  // namespace runtime
