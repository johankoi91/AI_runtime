#pragma once

#include "runtime/backend.h"

namespace runtime {

class CPUBackend final : public Backend {
 public:
  std::string_view name() const override;
  Tensor add(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const override;
  Tensor matmul(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const override;
};

}  // namespace runtime
