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
  Tensor sigmoid(const Tensor& input, MemoryPool& pool) const override;
  Tensor tanh(const Tensor& input, MemoryPool& pool) const override;
  Tensor relu(const Tensor& input, MemoryPool& pool) const override;
  Tensor conv2d(
      const Tensor& input,
      const Tensor& weights,
      const Tensor& bias,
      std::size_t stride,
      std::size_t padding,
      MemoryPool& pool) const override;
  Tensor gru_step(
      const Tensor& input,
      const Tensor& hidden,
      const Tensor& input_weights,
      const Tensor& hidden_weights,
      const Tensor& bias,
      MemoryPool& pool) const override;
};

}  // namespace runtime
