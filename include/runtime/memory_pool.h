#pragma once

#include <cstddef>
#include <vector>

#include "runtime/storage.h"
#include "runtime/tensor.h"

namespace runtime {

// MemoryPool 是一个非常简化的内存池。
//
// 为什么 runtime 喜欢内存池？
// 因为如果每次算子执行都单独 malloc/free：
// - 会慢
// - 会产生碎片
// - 生命周期管理会很麻烦
//
// 这一版实现的是最容易理解的 arena 风格：
// - 先申请一个 chunk
// - 之后在 chunk 里顺序切片
// - reset 时不释放 chunk，只把 offset 归零，准备复用
class MemoryPool {
 public:
  explicit MemoryPool(
      std::size_t default_chunk_size = 1 << 20,
      std::size_t default_alignment = 64);

  // 从 pool 里申请一段指定字节数的内存。
  Buffer allocate(
      std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t));

  // 直接申请一个 tensor 所需的底层内存。
  Tensor allocate_tensor(const Shape& shape, DType dtype = DType::Float32);

  // 重置所有 chunk 的 offset，让下一轮分配复用旧内存。
  void reset();

  std::size_t used_bytes() const { return used_bytes_; }
  std::size_t peak_bytes() const { return peak_bytes_; }
  std::size_t allocation_count() const { return allocation_count_; }
  std::size_t chunk_count() const { return chunks_.size(); }

 private:
  // Chunk 表示内存池内部的一大块连续内存。
  struct Chunk {
    Buffer backing;
    std::size_t capacity = 0;
    std::size_t offset = 0;
  };

  // 把一个偏移量向上对齐到 alignment 的整数倍。
  static std::size_t align_up(std::size_t value, std::size_t alignment);

  // 找到一个能放下本次分配请求的 chunk，如果没有就新建一个。
  Chunk& reserve_chunk(std::size_t min_bytes, std::size_t alignment);

  std::vector<Chunk> chunks_;
  std::size_t default_chunk_size_;
  std::size_t default_alignment_;
  std::size_t used_bytes_ = 0;
  std::size_t peak_bytes_ = 0;
  std::size_t allocation_count_ = 0;
};

}  // namespace runtime
