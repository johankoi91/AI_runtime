#include "runtime/tensor.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace runtime {

namespace {

std::size_t product(const Shape& shape) {
  return std::accumulate(
      shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{});
}

}  // namespace

Tensor::Tensor(
    DType dtype,
    Shape shape,
    Buffer buffer,
    Strides strides,
    std::size_t offset_elements)
    : dtype_(dtype),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      buffer_(std::move(buffer)),
      offset_elements_(offset_elements) {
  if (strides_.empty()) {
    strides_ = make_contiguous_strides(shape_);
  }
  validate();
}

Tensor Tensor::empty(const Shape& shape, DType dtype) {
  const auto bytes = product(shape) * dtype_size(dtype);
  return Tensor(dtype, shape, make_owned_buffer(bytes));
}

Tensor Tensor::zeros(const Shape& shape, DType dtype) {
  Tensor tensor = empty(shape, dtype);
  if (dtype == DType::Float32) {
    std::fill_n(tensor.data<float>(), tensor.numel(), 0.0f);
  }
  return tensor;
}

Tensor Tensor::from_vector(const Shape& shape, const std::vector<float>& values) {
  Tensor tensor = empty(shape, DType::Float32);
  if (values.size() != tensor.numel()) {
    throw std::runtime_error("from_vector: value count does not match shape");
  }
  std::copy(values.begin(), values.end(), tensor.data<float>());
  return tensor;
}

std::size_t Tensor::numel() const {
  return product(shape_);
}

std::size_t Tensor::nbytes() const {
  return numel() * dtype_size(dtype_);
}

bool Tensor::is_contiguous() const {
  return strides_ == make_contiguous_strides(shape_);
}

Tensor Tensor::view(
    const Shape& shape,
    const Strides& strides,
    std::size_t offset_elements) const {
  return Tensor(dtype_, shape, buffer_, strides, offset_elements_ + offset_elements);
}

void Tensor::fill(float value) {
  if (dtype_ != DType::Float32) {
    throw std::runtime_error("fill only supports float32");
  }
  std::fill_n(data<float>(), numel(), value);
}

std::string Tensor::debug_string() const {
  std::ostringstream stream;
  stream << "Tensor(dtype=" << dtype_name(dtype_) << ", shape=[";
  for (std::size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << shape_[i];
  }
  stream << "], strides=[";
  for (std::size_t i = 0; i < strides_.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << strides_[i];
  }
  stream << "], contiguous=" << (is_contiguous() ? "true" : "false") << ")";
  return stream.str();
}

Strides Tensor::make_contiguous_strides(const Shape& shape) {
  Strides strides(shape.size(), 1);
  if (shape.empty()) {
    return strides;
  }

  for (std::size_t i = shape.size(); i-- > 0;) {
    if (i + 1 < shape.size()) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  }
  return strides;
}

void Tensor::validate() const {
  if (shape_.size() != strides_.size()) {
    throw std::runtime_error("tensor validate: shape and strides size mismatch");
  }

  const auto required_bytes =
      (offset_elements_ + numel()) * dtype_size(dtype_);

  if (buffer_ && required_bytes > buffer_.bytes) {
    throw std::runtime_error("tensor validate: buffer too small for tensor");
  }
}

}  // namespace runtime
