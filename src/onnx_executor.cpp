#include "runtime/onnx_executor.h"

#include <numeric>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <iostream>
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
    if (attribute.integer != 0) return {attribute.integer};
    return fallback;
  } catch (const std::out_of_range&) {
    return fallback;
  }
}

std::vector<std::int64_t> tensor_ints(const Tensor& tensor) {
  if (tensor.dtype() != DType::Float32 || !tensor.is_contiguous()) {
    throw std::runtime_error("ONNX shape input must be contiguous float32");
  }
  std::vector<std::int64_t> values;
  for (std::size_t index = 0; index < tensor.numel(); ++index) {
    values.push_back(static_cast<std::int64_t>(tensor.data<float>()[index]));
  }
  return values;
}

Tensor copy_with_shape(const Tensor& input, const Shape& shape) {
  if (product(shape) != input.numel()) throw std::runtime_error("ONNX shape element count mismatch");
  return cpu::reshape(input, shape);
}

Tensor contiguous_copy(const Tensor& input, MemoryPool& pool) {
  if (input.is_contiguous()) return input;
  Tensor output = pool.allocate_tensor(input.shape(), input.dtype());
  std::vector<std::size_t> index(input.rank());
  for (std::size_t linear = 0; linear < input.numel(); ++linear) {
    std::size_t remaining = linear;
    std::size_t source_offset = input.offset_elements();
    for (std::size_t dimension = input.rank(); dimension-- > 0;) {
      index[dimension] = remaining % input.shape()[dimension];
      remaining /= input.shape()[dimension];
      source_offset += index[dimension] * input.strides()[dimension];
    }
    output.data<float>()[linear] = input.data<float>()[source_offset - input.offset_elements()];
  }
  return output;
}

Tensor pad_tensor(const Tensor& input, const std::vector<std::int64_t>& pads, float value, MemoryPool& pool) {
  if ((pads.size() < input.rank() * 2 || pads.size() % 2 != 0) || !input.is_contiguous()) {
    throw std::runtime_error("ONNX Pad requires compatible rank and pads");
  }
  const std::size_t pad_rank = pads.size() / 2;
  const std::size_t leading = pad_rank - input.rank();
  for (std::size_t index = 0; index < leading; ++index) {
    if (pads[index] != 0 || pads[index + pad_rank] != 0) throw std::runtime_error("Pad cannot remove leading dimensions");
  }
  Shape output_shape(input.shape());
  for (std::size_t index = 0; index < input.rank(); ++index) {
    const std::size_t pad_index = leading + index;
    if (pads[pad_index] < 0 || pads[pad_index + pad_rank] < 0) throw std::runtime_error("negative Pad is unsupported");
    output_shape[index] += static_cast<std::size_t>(pads[pad_index] + pads[pad_index + pad_rank]);
  }
  Tensor output = pool.allocate_tensor(output_shape);
  output.fill(value);
  std::vector<std::size_t> index(input.rank());
  for (std::size_t linear = 0; linear < input.numel(); ++linear) {
    std::size_t remaining = linear;
    std::size_t destination = 0;
    for (std::size_t dimension = input.rank(); dimension-- > 0;) {
      index[dimension] = remaining % input.shape()[dimension];
      remaining /= input.shape()[dimension];
      const std::size_t coordinate = index[dimension] + static_cast<std::size_t>(pads[leading + dimension]);
      destination = coordinate + destination * output_shape[dimension];
    }
    output.data<float>()[destination] = input.data<float>()[linear];
  }
  return output;
}

Tensor slice_tensor(
    const Tensor& input,
    const std::vector<std::int64_t>& starts,
    const std::vector<std::int64_t>& ends,
    const std::vector<std::int64_t>& axes,
    const std::vector<std::int64_t>& steps,
    MemoryPool& pool) {
  if (!input.is_contiguous() || starts.size() != ends.size() || starts.size() != axes.size() ||
      starts.size() != steps.size()) throw std::runtime_error("Slice arguments have mismatched lengths");
  Shape output_shape = input.shape();
  std::vector<std::int64_t> begin(input.rank(), 0);
  std::vector<std::int64_t> stride(input.rank(), 1);
  for (std::size_t index = 0; index < axes.size(); ++index) {
    const std::int64_t normalized_axis = axes[index] < 0 ? axes[index] + input.rank() : axes[index];
    if (normalized_axis < 0 || static_cast<std::size_t>(normalized_axis) >= input.rank() || steps[index] == 0) {
      throw std::runtime_error("Slice axis or step is invalid");
    }
    const std::size_t axis = static_cast<std::size_t>(normalized_axis);
    const std::int64_t dimension = static_cast<std::int64_t>(input.shape()[axis]);
    const std::int64_t step_value = steps[index];
    auto normalize = [&](std::int64_t value, bool is_start) {
      if (value < 0) value += dimension;
      if (step_value > 0) return std::max<std::int64_t>(0, std::min(value, dimension));
      if (is_start) return std::max<std::int64_t>(0, std::min(value, dimension - 1));
      return std::max<std::int64_t>(-1, std::min(value, dimension - 1));
    };
    const std::int64_t start = normalize(starts[index], true);
    const std::int64_t end = normalize(ends[index], false);
    begin[axis] = start;
    stride[axis] = step_value;
    if (step_value > 0) {
      output_shape[axis] = end <= start ? 0 : static_cast<std::size_t>((end - start + step_value - 1) / step_value);
    } else {
      output_shape[axis] = start <= end ? 0 : static_cast<std::size_t>((start - end - 1) / (-step_value) + 1);
    }
  }
  if (std::find(output_shape.begin(), output_shape.end(), 0) != output_shape.end()) {
    throw std::runtime_error("empty Slice outputs are unsupported");
  }
  Tensor output = pool.allocate_tensor(output_shape);
  std::vector<std::size_t> output_index(output.rank());
  std::vector<std::size_t> input_strides(input.rank(), 1);
  for (std::size_t dimension = input.rank(); dimension-- > 0;) {
    if (dimension + 1 < input.rank()) input_strides[dimension] = input_strides[dimension + 1] * input.shape()[dimension + 1];
  }
  for (std::size_t linear = 0; linear < output.numel(); ++linear) {
    std::size_t remaining = linear;
    std::size_t input_offset = 0;
    for (std::size_t dimension = output.rank(); dimension-- > 0;) {
      output_index[dimension] = remaining % output_shape[dimension];
      remaining /= output_shape[dimension];
      const std::int64_t coordinate = begin[dimension] +
          static_cast<std::int64_t>(output_index[dimension]) * stride[dimension];
      if (coordinate < 0 || coordinate >= static_cast<std::int64_t>(input.shape()[dimension])) {
        throw std::runtime_error("Slice coordinate is out of range");
      }
      input_offset += static_cast<std::size_t>(coordinate) * input_strides[dimension];
    }
    output.data<float>()[linear] = input.data<float>()[input_offset];
  }
  return output;
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
    if (node.op_type == "Add" || node.op_type == "Mul" || node.op_type == "MatMul") {
      if (node.inputs.size() != 2) throw std::runtime_error("binary ONNX node requires two inputs");
      const Tensor& lhs = lookup(tensors, node.inputs[0]);
      const Tensor& rhs = lookup(tensors, node.inputs[1]);
      if (std::getenv("HUSH_DEBUG_SHAPES") != nullptr && node.op_type == "Add") {
        std::cerr << "[add] " << node.name << " lhs=" << lhs.debug_string()
                  << " rhs=" << rhs.debug_string() << "\n";
      }
      output = node.op_type == "Add" ? backend_.add(lhs, rhs, pool)
          : node.op_type == "Mul" ? backend_.mul(lhs, rhs, pool)
          : backend_.matmul(lhs, rhs, pool);
    } else if (node.op_type == "Relu" || node.op_type == "Sigmoid" || node.op_type == "Tanh") {
      if (node.inputs.size() != 1) throw std::runtime_error("activation node requires one input");
      const Tensor& input = lookup(tensors, node.inputs[0]);
      output = node.op_type == "Relu" ? backend_.relu(input, pool)
          : node.op_type == "Sigmoid" ? backend_.sigmoid(input, pool) : backend_.tanh(input, pool);
    } else if (node.op_type == "Conv") {
      if (node.inputs.size() < 2 || node.inputs.size() > 3) throw std::runtime_error("Conv input count unsupported");
      Tensor input = contiguous_copy(lookup(tensors, node.inputs[0]), pool);
      Tensor weights = contiguous_copy(lookup(tensors, node.inputs[1]), pool);
      Tensor bias = node.inputs.size() == 3 ? contiguous_copy(lookup(tensors, node.inputs[2]), pool) : Tensor::zeros({weights.shape()[0]});
      const auto strides = attribute_values(graph, node, "strides", {1, 1});
      const auto pads = attribute_values(graph, node, "pads", {0, 0, 0, 0});
      const auto dilations = attribute_values(graph, node, "dilations", {1, 1});
      const auto groups = attribute_values(graph, node, "group", {1});
      if (strides.size() != 2 || pads.size() != 4 || dilations.size() != 2 || groups.size() != 1 ||
          strides[0] <= 0 || strides[1] <= 0 || pads[0] < 0 || pads[1] < 0 ||
          pads[2] < 0 || pads[3] < 0 || dilations[0] != 1 || dilations[1] != 1 ||
          false) {
        throw std::runtime_error("ONNX Conv attributes unsupported: strides=" + std::to_string(strides.size()) +
            " pads=" + std::to_string(pads.size()) + " dilation0=" + std::to_string(dilations[0]) +
            " dilation1=" + std::to_string(dilations[1]) + " group=" + std::to_string(groups[0]) +
            " stride0=" + std::to_string(strides[0]) + " stride1=" + std::to_string(strides[1]) +
            " groups_size=" + std::to_string(groups.size()) +
            " pad=" + std::to_string(pads[0]) + "," + std::to_string(pads[1]) + "," +
            std::to_string(pads[2]) + "," + std::to_string(pads[3]));
      }
      if (pads[0] != pads[2] || pads[1] != pads[3]) {
        const std::vector<std::int64_t> full_pads = {
            0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]};
        input = pad_tensor(input, full_pads, 0.0f, pool);
      }
      const std::size_t symmetric_padding = pads[0] == pads[2] && pads[1] == pads[3]
          ? static_cast<std::size_t>(pads[0]) : 0;
      output = backend_.conv2d(input, weights, bias, static_cast<std::size_t>(strides[0]),
                               static_cast<std::size_t>(strides[1]),
                               symmetric_padding, static_cast<std::size_t>(groups[0]), pool);
    } else if (node.op_type == "ConvTranspose") {
      if (node.inputs.size() < 2 || node.inputs.size() > 3) {
        throw std::runtime_error("ConvTranspose input count unsupported");
      }
      Tensor input = contiguous_copy(lookup(tensors, node.inputs[0]), pool);
      Tensor weights = contiguous_copy(lookup(tensors, node.inputs[1]), pool);
      Tensor bias = node.inputs.size() == 3
          ? contiguous_copy(lookup(tensors, node.inputs[2]), pool)
          : Tensor::zeros({weights.shape()[1]});
      const auto strides = attribute_values(graph, node, "strides", {1, 1});
      const auto pads = attribute_values(graph, node, "pads", {0, 0, 0, 0});
      const auto dilations = attribute_values(graph, node, "dilations", {1, 1});
      const auto groups = attribute_values(graph, node, "group", {1});
      if (strides.size() != 2 || pads.size() != 4 || dilations.size() != 2 || groups.size() != 1 ||
          strides[0] <= 0 || strides[1] <= 0 || pads[0] < 0 || pads[1] < 0 ||
          pads[2] < 0 || pads[3] < 0 || dilations[0] != 1 || dilations[1] != 1 || groups[0] != 1 ||
          pads[0] != pads[2] || pads[1] != pads[3] || strides[0] != strides[1]) {
        throw std::runtime_error(
            "ONNX ConvTranspose attributes are outside the supported 2D symmetric subset");
      }
      output = backend_.conv_transpose2d(
          input, weights, bias, static_cast<std::size_t>(strides[0]),
          static_cast<std::size_t>(pads[0]), pool);
    } else if (node.op_type == "GRU") {
      output = run_gru_step(backend_, node, tensors, pool);
    } else if (node.op_type == "Reshape") {
      if (node.inputs.size() != 2) throw std::runtime_error("Reshape requires data and shape inputs");
      const Tensor& shape_tensor = lookup(tensors, node.inputs[1]);
      std::vector<std::size_t> shape;
      std::size_t inferred_index = std::numeric_limits<std::size_t>::max();
      std::size_t known_elements = 1;
      for (std::size_t index = 0; index < shape_tensor.numel(); ++index) {
        const float value = shape_tensor.data<float>()[index];
        if (value == -1.0f) {
          if (inferred_index != std::numeric_limits<std::size_t>::max()) throw std::runtime_error("Reshape allows only one inferred dimension");
          inferred_index = index;
          shape.push_back(1);
        } else if (value == 0.0f) {
          if (index >= lookup(tensors, node.inputs[0]).rank()) throw std::runtime_error("Reshape zero dimension is out of range");
          shape.push_back(lookup(tensors, node.inputs[0]).shape()[index]);
          known_elements *= shape.back();
        } else if (value > 0.0f) {
          shape.push_back(static_cast<std::size_t>(value));
          known_elements *= shape.back();
        } else {
          throw std::runtime_error("Reshape has invalid negative dimension");
        }
      }
      if (inferred_index != std::numeric_limits<std::size_t>::max()) {
        if (known_elements == 0 || lookup(tensors, node.inputs[0]).numel() % known_elements != 0) {
          throw std::runtime_error("Reshape cannot infer dimension");
        }
        shape[inferred_index] = lookup(tensors, node.inputs[0]).numel() / known_elements;
      }
      const Tensor& reshape_input = lookup(tensors, node.inputs[0]);
      if (product(shape) != reshape_input.numel() && shape.size() == reshape_input.rank() + 1 &&
          shape.back() > 1 && std::equal(reshape_input.shape().begin(), reshape_input.shape().end(),
                                          shape.begin())) {
        Tensor expanded = pool.allocate_tensor(shape);
        for (std::size_t index = 0; index < expanded.numel(); ++index) {
          expanded.data<float>()[index] = reshape_input.data<float>()[index / shape.back()];
        }
        output = std::move(expanded);
        tensors[node.outputs[0]] = std::move(output);
        continue;
      }
      try {
        output = cpu::reshape(lookup(tensors, node.inputs[0]), shape);
      } catch (const std::exception& error) {
        std::string shape_text;
        for (std::size_t value : shape) shape_text += std::to_string(value) + ",";
        throw std::runtime_error("Reshape node " + node.name + " input=" + node.inputs[0] +
                                 " shape=" + shape_text + ": " + error.what());
      }
    } else if (node.op_type == "Transpose") {
      std::vector<std::size_t> permutation;
      try {
        const auto& attribute = graph.attribute(node, "perm");
        for (std::int64_t value : attribute.integers) permutation.push_back(static_cast<std::size_t>(value));
        if (permutation.empty()) {
          for (std::size_t index = lookup(tensors, node.inputs[0]).rank(); index-- > 0;) permutation.push_back(index);
        }
      } catch (const std::out_of_range&) {
        for (std::size_t index = lookup(tensors, node.inputs[0]).rank(); index-- > 0;) permutation.push_back(index);
      }
      output = cpu::transpose(lookup(tensors, node.inputs[0]), permutation, &pool);
    } else if (node.op_type == "Concat") {
      const auto& attribute = graph.attribute(node, "axis");
      std::vector<const Tensor*> concat_inputs;
      std::vector<Tensor> concat_adapters;
      concat_adapters.reserve(node.inputs.size());
      const Tensor& first_tensor = lookup(tensors, node.inputs[0]);
      for (const std::string& input_name : node.inputs) {
        const Tensor* tensor = &lookup(tensors, input_name);
        if (tensor->rank() < first_tensor.rank() && first_tensor.rank() == 2) {
          const std::size_t axis = static_cast<std::size_t>(attribute.integer);
          const std::size_t length = tensor->numel();
          const Tensor contiguous = contiguous_copy(*tensor, pool);
          concat_adapters.push_back(cpu::reshape(contiguous, axis == 0
              ? Shape{length, 1} : Shape{1, length}));
          tensor = &concat_adapters.back();
        }
        if (tensor->rank() != first_tensor.rank()) {
          throw std::runtime_error("Concat node " + node.name + " input " + input_name +
                                   " rank=" + std::to_string(tensor->rank()) +
                                   " expected=" + std::to_string(first_tensor.rank()));
        }
        concat_inputs.push_back(tensor);
      }
      try {
        output = cpu::concat(concat_inputs, static_cast<std::size_t>(attribute.integer), &pool);
      } catch (const std::exception& error) {
        throw std::runtime_error("Concat node " + node.name + ": " + error.what());
      }
    } else if (node.op_type == "Flatten") {
      if (node.inputs.size() != 1) throw std::runtime_error("Flatten requires one input");
      const Tensor& input = lookup(tensors, node.inputs[0]);
      const std::size_t axis = attribute_size(graph, node, "axis", 1);
      if (axis > input.rank()) throw std::runtime_error("Flatten axis is out of range");
      const std::size_t left = product(Shape(input.shape().begin(), input.shape().begin() + axis));
      const std::size_t right = input.numel() / left;
      output = cpu::reshape(input, {left, right});
    } else if (node.op_type == "Shape") {
      if (node.inputs.size() != 1) throw std::runtime_error("Shape requires one input");
      const Tensor& input = lookup(tensors, node.inputs[0]);
      std::vector<float> shape_values;
      shape_values.reserve(input.rank());
      for (std::size_t dimension : input.shape()) {
        shape_values.push_back(static_cast<float>(dimension));
      }
      output = Tensor::from_vector({shape_values.size()}, shape_values);
    } else if (node.op_type == "Constant") {
      if (!node.inputs.empty()) throw std::runtime_error("Constant cannot have inputs");
      const auto& value = graph.attribute(node, "value");
      if (!value.has_tensor) throw std::runtime_error("Constant requires a tensor value attribute");
      output = Tensor::from_vector(value.tensor.shape, value.tensor.values);
    } else if (node.op_type == "ConstantOfShape") {
      if (node.inputs.size() != 1) throw std::runtime_error("ConstantOfShape requires a shape input");
      const auto shape_values = tensor_ints(lookup(tensors, node.inputs[0]));
      Shape shape;
      for (std::int64_t value : shape_values) {
        if (value < 0) throw std::runtime_error("ConstantOfShape has a negative dimension");
        shape.push_back(static_cast<std::size_t>(value));
      }
      float fill_value = 0.0f;
      try {
        const auto& value = graph.attribute(node, "value");
        if (value.has_tensor && !value.tensor.values.empty()) fill_value = value.tensor.values[0];
      } catch (const std::out_of_range&) {
      }
      output = Tensor::zeros(shape);
      output.fill(fill_value);
    } else if (node.op_type == "Cast") {
      if (node.inputs.size() != 1) throw std::runtime_error("Cast requires one input");
      const auto& attribute = graph.attribute(node, "to");
      if (attribute.integer != 1 && attribute.integer != 6 && attribute.integer != 7 && attribute.integer != 10) {
        throw std::runtime_error("unsupported ONNX Cast target type");
      }
      output = lookup(tensors, node.inputs[0]);
    } else if (node.op_type == "Unsqueeze") {
      const Tensor& input = lookup(tensors, node.inputs[0]);
      const std::vector<std::int64_t> axes = node.inputs.size() > 1
          ? tensor_ints(lookup(tensors, node.inputs[1]))
          : attribute_values(graph, node, "axes", {});
      Shape shape = input.shape();
      for (std::int64_t axis : axes) {
        if (axis < 0 || static_cast<std::size_t>(axis) > shape.size()) throw std::runtime_error("Unsqueeze axis is invalid");
        shape.insert(shape.begin() + axis, 1);
      }
      output = copy_with_shape(input, shape);
    } else if (node.op_type == "Squeeze") {
      const Tensor& input = lookup(tensors, node.inputs[0]);
      const std::vector<std::int64_t> axes = node.inputs.size() > 1
          ? tensor_ints(lookup(tensors, node.inputs[1]))
          : attribute_values(graph, node, "axes", {});
      Shape shape;
      for (std::size_t dimension = 0; dimension < input.rank(); ++dimension) {
        const bool remove = std::find(axes.begin(), axes.end(), static_cast<std::int64_t>(dimension)) != axes.end();
        if (remove && input.shape()[dimension] != 1) throw std::runtime_error("Squeeze axis is not one");
        if (!remove) shape.push_back(input.shape()[dimension]);
      }
      output = copy_with_shape(input, shape);
    } else if (node.op_type == "Gather") {
      if (node.inputs.size() != 2) throw std::runtime_error("Gather requires data and indices");
      const Tensor& input = lookup(tensors, node.inputs[0]);
      const auto indices = tensor_ints(lookup(tensors, node.inputs[1]));
      const std::size_t axis = attribute_size(graph, node, "axis", 0);
      if (axis >= input.rank() || axis != 0) throw std::runtime_error("Gather currently supports axis 0 only");
      const std::size_t block = input.numel() / input.shape()[0];
      const bool scalar_index = lookup(tensors, node.inputs[1]).rank() == 0;
      Tensor gathered = pool.allocate_tensor(scalar_index ? Shape{block} : Shape{indices.size(), block});
      for (std::size_t index = 0; index < indices.size(); ++index) {
        if (indices[index] < 0 || static_cast<std::size_t>(indices[index]) >= input.shape()[0]) throw std::runtime_error("Gather index out of range");
        std::copy_n(input.data<float>() + static_cast<std::size_t>(indices[index]) * block,
                    block, gathered.data<float>() + (scalar_index ? 0 : index * block));
      }
      Shape shape = input.shape();
      if (scalar_index) {
        shape.erase(shape.begin());
        output = copy_with_shape(gathered, shape);
      } else {
        shape[0] = indices.size();
        output = copy_with_shape(gathered, shape);
      }
    } else if (node.op_type == "Pad") {
      if (node.inputs.size() < 2 || node.inputs.size() > 3) throw std::runtime_error("Pad input count unsupported");
      const auto pads = tensor_ints(lookup(tensors, node.inputs[1]));
      float value = 0.0f;
      if (node.inputs.size() == 3) value = lookup(tensors, node.inputs[2]).data<float>()[0];
      output = pad_tensor(lookup(tensors, node.inputs[0]), pads, value, pool);
    } else if (node.op_type == "Slice") {
      if (node.inputs.size() < 3 || node.inputs.size() > 5) throw std::runtime_error("Slice input count unsupported");
      const auto starts = tensor_ints(lookup(tensors, node.inputs[1]));
      const auto ends = tensor_ints(lookup(tensors, node.inputs[2]));
      const auto axes = node.inputs.size() > 3 ? tensor_ints(lookup(tensors, node.inputs[3])) : std::vector<std::int64_t>{0};
      const auto steps = node.inputs.size() > 4 ? tensor_ints(lookup(tensors, node.inputs[4])) : std::vector<std::int64_t>{1};
      output = slice_tensor(lookup(tensors, node.inputs[0]), starts, ends, axes, steps, pool);
    } else if (node.op_type == "Einsum") {
      if (node.inputs.size() != 2) throw std::runtime_error("Einsum requires two inputs");
      const auto& equation = graph.attribute(node, "equation").string;
      const Tensor& lhs = lookup(tensors, node.inputs[0]);
      const Tensor& rhs = lookup(tensors, node.inputs[1]);
      if (std::getenv("HUSH_DEBUG_SHAPES") != nullptr) {
        std::cerr << "[einsum] " << node.name << " equation=" << equation
                  << " lhs=" << lhs.debug_string() << " rhs=" << rhs.debug_string() << "\n";
      }
      if (equation == "ij,jk->ik") {
        output = backend_.matmul(lhs, rhs, pool);
      } else if (equation == "btgi,gih->btgh") {
        if (lhs.rank() != 4 || rhs.rank() != 3 || lhs.shape()[2] != rhs.shape()[0] ||
            lhs.shape()[3] != rhs.shape()[1]) throw std::runtime_error("Einsum btgi,gih->btgh shape mismatch");
        const Shape output_shape = {lhs.shape()[0], lhs.shape()[1], lhs.shape()[2], rhs.shape()[2]};
        output = pool.allocate_tensor(output_shape);
        for (std::size_t batch = 0; batch < lhs.shape()[0]; ++batch) {
          for (std::size_t time = 0; time < lhs.shape()[1]; ++time) {
            for (std::size_t group = 0; group < lhs.shape()[2]; ++group) {
              for (std::size_t hidden = 0; hidden < rhs.shape()[2]; ++hidden) {
                float sum = 0.0f;
                for (std::size_t input_index = 0; input_index < lhs.shape()[3]; ++input_index) {
                  const std::size_t lhs_offset = ((batch * lhs.shape()[1] + time) * lhs.shape()[2] + group) * lhs.shape()[3] + input_index;
                  const std::size_t rhs_offset = (group * rhs.shape()[1] + input_index) * rhs.shape()[2] + hidden;
                  sum += lhs.data<float>()[lhs_offset] * rhs.data<float>()[rhs_offset];
                }
                const std::size_t output_offset = ((batch * output_shape[1] + time) * output_shape[2] + group) * output_shape[3] + hidden;
                output.data<float>()[output_offset] = sum;
              }
            }
          }
        }
      } else if (equation == "i,ij->j") {
        if (lhs.rank() != 1 || rhs.rank() != 2 || lhs.shape()[0] != rhs.shape()[0]) {
          throw std::runtime_error("Einsum i,ij->j shape mismatch");
        }
        Tensor lhs_matrix = cpu::reshape(lhs, {1, lhs.shape()[0]});
        output = backend_.matmul(lhs_matrix, rhs, pool);
        output = cpu::reshape(output, {rhs.shape()[1]});
      } else {
        throw std::runtime_error("unsupported Einsum equation: " + equation);
      }
    } else {
      throw std::runtime_error("unsupported ONNX op in stage 15: " + node.op_type);
    }
    tensors[node.outputs[0]] = std::move(output);
    if (std::getenv("HUSH_DEBUG_SHAPES") != nullptr) {
      std::cerr << "[shape] " << node.name << " " << node.op_type << " -> "
                << tensors[node.outputs[0]].debug_string() << "\n";
    }
  }
  return tensors;
}

}  // namespace runtime
