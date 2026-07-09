#pragma once

#include "runtime/memory_pool.h"
#include "runtime/tensor.h"

namespace runtime::cpu {

Tensor add(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool = nullptr);
Tensor matmul(const Tensor& lhs, const Tensor& rhs, MemoryPool* pool = nullptr);

}  // namespace runtime::cpu
