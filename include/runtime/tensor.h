#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

#include "runtime/dtype.h"
#include "runtime/storage.h"

namespace runtime {

// Shape 表示每个维度有多大。
// 例如一个 2x3 的矩阵，shape 就是 {2, 3}。
using Shape = std::vector<std::size_t>;

// Strides 表示“沿着每个维度前进一步，需要跳过多少个元素”。
// 例如 contiguous 的 2x3 矩阵，stride 通常是 {3, 1}。
// 这意味着：
// - 行索引 +1，要跳过 3 个元素
// - 列索引 +1，要跳过 1 个元素
using Strides = std::vector<std::size_t>;

// Tensor 是整个 runtime 最重要的数据结构之一。
//
// 你可以把它理解成：
// “一块原始内存 Buffer + 如何解释这块内存的元信息”
//
// 这些元信息主要包括：
// - dtype：每个元素是什么类型
// - shape：这是一个几维张量，每维多大
// - strides：怎样从一个元素跳到另一个元素
// - offset：这个 Tensor view 从底层 Buffer 的第几个元素开始
//
// 现在这版 Tensor 先支持最核心的能力，方便我们逐层理解 runtime。
class Tensor {
 public:
  Tensor() = default;
  Tensor(
      DType dtype,
      Shape shape,
      Buffer buffer,
      Strides strides = {},
      std::size_t offset_elements = 0);

  // 创建一块未初始化的 tensor。
  static Tensor empty(const Shape& shape, DType dtype = DType::Float32);

  // 创建并清零。
  static Tensor zeros(const Shape& shape, DType dtype = DType::Float32);

  // 从一组 float 数据直接构造 tensor，适合 demo 和测试。
  static Tensor from_vector(const Shape& shape, const std::vector<float>& values);

  const Shape& shape() const { return shape_; }
  const Strides& strides() const { return strides_; }
  DType dtype() const { return dtype_; }
  std::size_t rank() const { return shape_.size(); }
  std::size_t offset_elements() const { return offset_elements_; }

  // 总元素个数。
  std::size_t numel() const;

  // 这个 tensor 逻辑上占多少字节。
  std::size_t nbytes() const;

  // 判断当前 layout 是否是标准连续布局。
  bool is_contiguous() const;

  template <typename T>
  T* data() {
    // data<T>() 会把“底层字节地址 + 元素偏移”转换成真正的类型化指针。
    return reinterpret_cast<T*>(buffer_.data + (offset_elements_ * dtype_size(dtype_)));
  }

  template <typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(
        buffer_.data + (offset_elements_ * dtype_size(dtype_)));
  }

  // 在同一块底层内存上创建一个 view。
  //
  // 重点是：view 不复制数据，它只是换了一种“看待同一块内存”的方式。
  // 这就是 stride / offset 的价值所在。
  Tensor view(
      const Shape& shape,
      const Strides& strides,
      std::size_t offset_elements = 0) const;

  // 把整个 tensor 都填成同一个值。
  void fill(float value);

  // 用于调试时打印关键信息。
  std::string debug_string() const;

 private:
  // 给一个 shape 计算出默认的 contiguous strides。
  static Strides make_contiguous_strides(const Shape& shape);

  // 在构造时做最基本的一致性检查。
  void validate() const;

  DType dtype_ = DType::Float32;
  Shape shape_;
  Strides strides_;

  // 真正存数据的地方在这里。
  Buffer buffer_;

  // 这个 Tensor 从底层 Buffer 的第几个“元素”开始看，而不是第几个字节。
  std::size_t offset_elements_ = 0;
};

}  // namespace runtime
