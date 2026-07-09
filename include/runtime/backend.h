#pragma once

#include <string_view>

#include "runtime/memory_pool.h"
#include "runtime/tensor.h"

namespace runtime {

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
