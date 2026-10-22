#pragma once

#include <string_view>

#include "runtime/memory_pool.h"
#include "runtime/trace.h"
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
  virtual Tensor mul(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const = 0;
  virtual Tensor matmul(
      const Tensor& lhs,
      const Tensor& rhs,
      MemoryPool& pool) const = 0;
  virtual Tensor sigmoid(const Tensor& input, MemoryPool& pool) const = 0;
  virtual Tensor tanh(const Tensor& input, MemoryPool& pool) const = 0;
  virtual Tensor relu(const Tensor& input, MemoryPool& pool) const = 0;
  virtual Tensor conv2d(
      const Tensor& input,
      const Tensor& weights,
      const Tensor& bias,
      std::size_t stride_h,
      std::size_t stride_w,
      std::size_t padding_h,
      std::size_t padding_w,
      std::size_t groups,
      MemoryPool& pool) const = 0;
  virtual Tensor conv_transpose2d(
      const Tensor& input,
      const Tensor& weights,
      const Tensor& bias,
      std::size_t stride_h,
      std::size_t stride_w,
      std::size_t padding_h,
      std::size_t padding_w,
      std::size_t groups,
      MemoryPool& pool) const = 0;
  virtual Tensor stft(
      const Tensor& samples,
      std::size_t fft_size,
      std::size_t hop_size,
      MemoryPool& pool) const = 0;
  virtual Tensor erb_filterbank(
      const Tensor& spectrum,
      const Tensor& filterbank,
      MemoryPool& pool) const = 0;
  virtual Tensor istft(
      const Tensor& spectrum,
      std::size_t fft_size,
      std::size_t hop_size,
      std::size_t original_samples,
      MemoryPool& pool) const = 0;
  virtual Tensor gru_step(
      const Tensor& input,
      const Tensor& hidden,
      const Tensor& input_weights,
      const Tensor& hidden_weights,
      const Tensor& bias,
      MemoryPool& pool) const = 0;
  virtual Tensor gru_sequence(
      const Tensor& sequence,
      const Tensor& initial_hidden,
      const Tensor& input_weights,
      const Tensor& hidden_weights,
      const Tensor& bias,
      MemoryPool& pool) const = 0;

  virtual void set_trace_sink(TraceSink* sink) const { (void)sink; }
};

}  // namespace runtime
