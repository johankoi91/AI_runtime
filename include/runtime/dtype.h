#pragma once

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace runtime {

// DType 是 runtime 里“元素类型”的最小抽象。
//
// 现在我们只支持 float32，是为了把注意力放在 tensor layout、kernel、
// memory pool 和 graph execution 上，而不是一开始就被大量数据类型分散注意力。
// 以后如果要扩展到 int8 / float16 / bfloat16，通常就是从这里开始往外扩。
enum class DType {
  Float32,
};

// 给定一个 DType，返回单个元素占多少字节。
//
// 例如 float32 的一个元素是 4 字节，这个信息会被 Tensor 和 MemoryPool
// 用来计算 buffer 大小。
inline std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return sizeof(float);
  }
  throw std::runtime_error("unsupported dtype");
}

// 把 DType 转成人类可读的字符串，主要用于 debug 输出。
inline std::string_view dtype_name(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return "float32";
  }
  throw std::runtime_error("unsupported dtype");
}

}  // namespace runtime
