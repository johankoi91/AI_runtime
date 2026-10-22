#include "runtime/kernels.h"

#include <numeric>
#include <stdexcept>

namespace runtime::cpu {

namespace {

std::size_t product(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{});
}

Strides strides_for(const Shape& shape) {
  Strides strides(shape.size(), 1);
  for (std::size_t index = shape.size(); index-- > 0;) {
    if (index + 1 < shape.size()) strides[index] = strides[index + 1] * shape[index + 1];
  }
  return strides;
}

void decode_index(std::size_t linear, const Shape& shape, std::vector<std::size_t>& index) {
  index.resize(shape.size());
  for (std::size_t dimension = shape.size(); dimension-- > 0;) {
    index[dimension] = linear % shape[dimension];
    linear /= shape[dimension];
  }
}

}  // namespace

Tensor reshape(const Tensor& input, const Shape& shape) {
  if (input.dtype() != DType::Float32 || !input.is_contiguous() || product(shape) != input.numel()) {
    throw std::invalid_argument("reshape requires a contiguous tensor with matching element count");
  }
  return input.view(shape, strides_for(shape));
}

Tensor transpose(const Tensor& input, const std::vector<std::size_t>& permutation, MemoryPool* pool) {
  if (input.dtype() != DType::Float32 || !input.is_contiguous() ||
      permutation.size() != input.rank()) {
    throw std::invalid_argument("transpose requires a contiguous float32 tensor and full permutation");
  }
  Shape output_shape(input.rank());
  std::vector<bool> used(input.rank(), false);
  for (std::size_t index = 0; index < permutation.size(); ++index) {
    if (permutation[index] >= input.rank() || used[permutation[index]]) {
      throw std::invalid_argument("transpose permutation is invalid");
    }
    used[permutation[index]] = true;
    output_shape[index] = input.shape()[permutation[index]];
  }
  Tensor output = pool == nullptr ? Tensor::empty(output_shape) : pool->allocate_tensor(output_shape);
  const Strides input_strides = strides_for(input.shape());
  for (std::size_t linear = 0; linear < output.numel(); ++linear) {
    std::vector<std::size_t> output_index;
    decode_index(linear, output_shape, output_index);
    std::size_t input_offset = 0;
    for (std::size_t index = 0; index < permutation.size(); ++index) {
      input_offset += output_index[index] * input_strides[permutation[index]];
    }
    output.data<float>()[linear] = input.data<float>()[input_offset];
  }
  return output;
}

Tensor concat(const std::vector<const Tensor*>& inputs, std::size_t axis, MemoryPool* pool) {
  if (inputs.empty() || inputs.front() == nullptr) throw std::invalid_argument("concat needs inputs");
  const Tensor& first = *inputs.front();
  if (first.dtype() != DType::Float32 || !first.is_contiguous() || axis >= first.rank()) {
    throw std::invalid_argument("concat has invalid first input");
  }
  Shape output_shape = first.shape();
  output_shape[axis] = 0;
  for (const Tensor* input : inputs) {
    if (input == nullptr || input->dtype() != DType::Float32 || !input->is_contiguous() ||
        input->rank() != first.rank()) throw std::invalid_argument("concat input is invalid");
    for (std::size_t dimension = 0; dimension < first.rank(); ++dimension) {
      if (dimension != axis && input->shape()[dimension] != first.shape()[dimension]) {
        throw std::invalid_argument("concat non-axis dimensions must match");
      }
    }
    output_shape[axis] += input->shape()[axis];
  }
  Tensor output = pool == nullptr ? Tensor::empty(output_shape) : pool->allocate_tensor(output_shape);
  const std::size_t outer = product(Shape(output_shape.begin(), output_shape.begin() + axis));
  const std::size_t inner = product(Shape(output_shape.begin() + axis + 1, output_shape.end()));
  std::size_t output_offset = 0;
  for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (const Tensor* input : inputs) {
      const std::size_t block = input->shape()[axis] * inner;
      const float* source = input->data<float>() + outer_index * block;
      std::copy(source, source + block, output.data<float>() + output_offset);
      output_offset += block;
    }
  }
  return output;
}

}  // namespace runtime::cpu
