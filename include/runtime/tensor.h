#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

#include "runtime/dtype.h"
#include "runtime/storage.h"

namespace runtime {

using Shape = std::vector<std::size_t>;
using Strides = std::vector<std::size_t>;

class Tensor {
 public:
  Tensor() = default;
  Tensor(
      DType dtype,
      Shape shape,
      Buffer buffer,
      Strides strides = {},
      std::size_t offset_elements = 0);

  static Tensor empty(const Shape& shape, DType dtype = DType::Float32);
  static Tensor zeros(const Shape& shape, DType dtype = DType::Float32);
  static Tensor from_vector(const Shape& shape, const std::vector<float>& values);

  const Shape& shape() const { return shape_; }
  const Strides& strides() const { return strides_; }
  DType dtype() const { return dtype_; }
  std::size_t rank() const { return shape_.size(); }
  std::size_t offset_elements() const { return offset_elements_; }

  std::size_t numel() const;
  std::size_t nbytes() const;
  bool is_contiguous() const;

  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(buffer_.data + (offset_elements_ * dtype_size(dtype_)));
  }

  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(
        buffer_.data + (offset_elements_ * dtype_size(dtype_)));
  }

  Tensor view(
      const Shape& shape,
      const Strides& strides,
      std::size_t offset_elements = 0) const;

  void fill(float value);
  std::string debug_string() const;

 private:
  static Strides make_contiguous_strides(const Shape& shape);
  void validate() const;

  DType dtype_ = DType::Float32;
  Shape shape_;
  Strides strides_;
  Buffer buffer_;
  std::size_t offset_elements_ = 0;
};

}  // namespace runtime
