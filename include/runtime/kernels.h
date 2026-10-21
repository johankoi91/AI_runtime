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
Tensor matmul(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool = nullptr);
Tensor sigmoid(const Tensor& input, MemoryPool* pool = nullptr);
Tensor tanh(const Tensor& input, MemoryPool* pool = nullptr);

}  // namespace runtime::cpu
