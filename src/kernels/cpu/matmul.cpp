#include "runtime/kernels.h"

#include <stdexcept>

namespace runtime::cpu {

Tensor matmul(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool) {
  if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32) {
    throw std::runtime_error("cpu::matmul only supports float32");
  }

  // 这一版只支持最标准的二维矩阵乘法。
  if (lhs.rank() != 2 || rhs.rank() != 2) {
    throw std::runtime_error("cpu::matmul requires rank-2 tensors");
  }

  // 仍然先限制到 contiguous，这样索引公式会简单很多。
  if (!lhs.is_contiguous() || !rhs.is_contiguous()) {
    throw std::runtime_error("cpu::matmul currently requires contiguous tensors");
  }

  // lhs: [m, k]
  // rhs: [k, n]
  // out: [m, n]
  const std::size_t m = lhs.shape()[0];
  const std::size_t k = lhs.shape()[1];
  const std::size_t rhs_k = rhs.shape()[0];
  const std::size_t n = rhs.shape()[1];

  if (k != rhs_k) {
    throw std::runtime_error("cpu::matmul inner dimensions must match");
  }

  Tensor out = pool ? pool->allocate_tensor({m, n}, lhs.dtype())
                    : Tensor::zeros({m, n}, lhs.dtype());
  out.fill(0.0f);

  const float* lhs_data = lhs.data<float>();
  const float* rhs_data = rhs.data<float>();
  float* out_data = out.data<float>();

  // 这是最经典的三层循环矩阵乘法：
  //
  // i: 决定输出的第几行
  // p: 决定 lhs 当前取哪一列，同时也是 rhs 当前取哪一行
  // j: 决定输出的第几列
  //
  // out[i, j] += lhs[i, p] * rhs[p, j]
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t p = 0; p < k; ++p) {
      const float lhs_value = lhs_data[i * k + p];
      for (std::size_t j = 0; j < n; ++j) {
        out_data[i * n + j] += lhs_value * rhs_data[p * n + j];
      }
    }
  }

  return out;
}

}  // namespace runtime::cpu
