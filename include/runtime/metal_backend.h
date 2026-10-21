#pragma once

#include <memory>

#include "runtime/backend.h"

namespace runtime {

class MetalBackend final : public Backend {
 public:
  MetalBackend();
  ~MetalBackend() override;

  MetalBackend(const MetalBackend&) = delete;
  MetalBackend& operator=(const MetalBackend&) = delete;

  std::string_view name() const override;
  Tensor add(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const override;
  Tensor matmul(
      const Tensor& lhs,
      const Tensor& rhs,
      MemoryPool& pool) const override;
  void set_trace_sink(TraceSink* sink) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runtime
