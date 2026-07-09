#pragma once

#include <string_view>

#include "runtime/memory_pool.h"
#include "runtime/tensor.h"

namespace runtime {

// Backend 是“执行后端”的抽象接口。
//
// 你可以把它理解成：
// “同一个 op，未来可能有不同的实现方式”
//
// 例如以后可能会出现：
// - CPUBackend
// - CUDABackend
// - MetalBackend
//
// 这样 Executor 就不需要直接依赖某个具体平台的 kernel 实现。
class Backend {
 public:
  virtual ~Backend() = default;

  virtual std::string_view name() const = 0;
  virtual Tensor add(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const = 0;
  virtual Tensor matmul(
      const Tensor& lhs,
      const Tensor& rhs,
      MemoryPool& pool) const = 0;
};

}  // namespace runtime
