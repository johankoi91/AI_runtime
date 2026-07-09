#include "runtime/cpu_backend.h"

#include "runtime/kernels.h"

namespace runtime {

std::string_view CPUBackend::name() const {
  return "cpu";
}

Tensor CPUBackend::add(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const {
  return cpu::add(lhs, rhs, &pool);
}

Tensor CPUBackend::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MemoryPool& pool) const {
  return cpu::matmul(lhs, rhs, &pool);
}

}  // namespace runtime
