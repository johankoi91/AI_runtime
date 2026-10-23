#pragma once

#include "runtime/memory_pool.h"
#include "runtime/tensor.h"

namespace runtime::cpu {

// 这一层是真正做数值计算的 CPU kernel。
//
// 目前先只放最基础的两个：
// - add：逐元素相加
// - matmul：矩阵乘法
//
// 它们的共同模式是：
// 1. 检查输入是否合法
// 2. 申请输出 tensor
// 3. 执行真正的循环计算
Tensor add(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool = nullptr);
Tensor mul(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool = nullptr);
Tensor matmul(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool = nullptr);
Tensor sigmoid(const Tensor& input, MemoryPool* pool = nullptr);
Tensor tanh(const Tensor& input, MemoryPool* pool = nullptr);
Tensor gru_step(
    const Tensor& input,
    const Tensor& hidden,
    const Tensor& input_weights,
    const Tensor& hidden_weights,
    const Tensor& bias,
    MemoryPool* pool = nullptr);
Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t padding_h,
    std::size_t padding_w,
    std::size_t groups = 1,
    MemoryPool* pool = nullptr);
Tensor conv_transpose2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t padding_h,
    std::size_t padding_w,
    std::size_t groups = 1,
    MemoryPool* pool = nullptr,
    std::size_t output_padding_h = 0,
    std::size_t output_padding_w = 0);
Tensor erb_filterbank(
    const Tensor& spectrum,
    const Tensor& filterbank,
    MemoryPool* pool = nullptr);
Tensor exponential_unit_norm(
    const Tensor& input,
    float& running_energy,
    float alpha,
    MemoryPool* pool = nullptr);
Tensor apply_erb_mask(
    const Tensor& spectrum,
    const Tensor& erb_mask,
    const Tensor& erb_projection,
    MemoryPool* pool = nullptr);
Tensor istft(
    const Tensor& spectrum,
    std::size_t fft_size,
    std::size_t hop_size,
    std::size_t original_samples,
    MemoryPool* pool = nullptr);
Tensor reshape(const Tensor& input, const Shape& shape);
Tensor transpose(const Tensor& input, const std::vector<std::size_t>& permutation, MemoryPool* pool = nullptr);
Tensor concat(const std::vector<const Tensor*>& inputs, std::size_t axis, MemoryPool* pool = nullptr);

}  // namespace runtime::cpu
