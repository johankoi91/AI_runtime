#pragma once

#include "runtime/backend.h"

namespace runtime {

// CPUBackend 是 Backend 的第一个具体实现。
//
// 它自己并不直接写数值循环，而是把调用转发给 runtime::cpu 命名空间下的 kernel。
class CPUBackend final : public Backend {
 public:
  std::string_view name() const override;
  Tensor add(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const override;
  Tensor matmul(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const override;
};

}  // namespace runtime
