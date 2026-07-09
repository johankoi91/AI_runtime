#include "runtime/memory_pool.h"

#include <algorithm>
#include <numeric>

namespace runtime {

namespace {

// 和 tensor.cpp 一样，这里也需要知道某个 shape 一共有多少元素。
std::size_t product(const Shape& shape) {
  return std::accumulate(
      shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{});
}

}  // namespace

MemoryPool::MemoryPool(
    std::size_t default_chunk_size,
    std::size_t default_alignment)
    : default_chunk_size_(default_chunk_size),
      default_alignment_(default_alignment) {}

Buffer MemoryPool::allocate(std::size_t bytes, std::size_t alignment) {
  if (bytes == 0) {
    return {};
  }

  // 第一步：先找到一个能容纳这次请求的 chunk。
  Chunk& chunk = reserve_chunk(bytes, std::max(alignment, default_alignment_));

  // 第二步：在 chunk 内部做对齐。
  const std::size_t aligned_offset = align_up(chunk.offset, alignment);

  // consumed 记录这次分配真正“推进”了多少空间，
  // 包括 alignment 产生的空洞。
  const std::size_t consumed = (aligned_offset + bytes) - chunk.offset;
  chunk.offset = aligned_offset + bytes;

  // 维护简单统计信息，后面做 benchmark 时很有用。
  used_bytes_ += consumed;
  peak_bytes_ = std::max(peak_bytes_, used_bytes_);
  ++allocation_count_;

  return {
      .owner = chunk.backing.owner,
      .data = chunk.backing.data + aligned_offset,
      .bytes = bytes,
  };
}

Tensor MemoryPool::allocate_tensor(const Shape& shape, DType dtype) {
  // MemoryPool 并不理解 tensor 的数学意义，
  // 它只是根据 shape 和 dtype 算出需要多少字节，然后交给 Tensor 解释。
  const auto bytes = product(shape) * dtype_size(dtype);
  return Tensor(dtype, shape, allocate(bytes));
}

void MemoryPool::reset() {
  // reset 不释放 chunk，只是把 offset 归零。
  // 这就是“池化复用”的核心。
  for (Chunk& chunk : chunks_) {
    chunk.offset = 0;
  }
  used_bytes_ = 0;
}

std::size_t MemoryPool::align_up(std::size_t value, std::size_t alignment) {
  // 例如 value=66, alignment=64，那么结果应该是 128。
  const std::size_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

MemoryPool::Chunk& MemoryPool::reserve_chunk(
    std::size_t min_bytes,
    std::size_t alignment) {
  // 先尝试复用最后一个 chunk。
  if (!chunks_.empty()) {
    Chunk& current = chunks_.back();
    const std::size_t aligned_offset = align_up(current.offset, alignment);
    if ((aligned_offset + min_bytes) <= current.capacity) {
      return current;
    }
  }

  const std::size_t capacity =
      std::max(default_chunk_size_, min_bytes + alignment);

  // 如果复用不了，就新建一个 chunk。
  chunks_.push_back(Chunk{
      .backing = make_owned_buffer(capacity, alignment),
      .capacity = capacity,
      .offset = 0,
  });
  return chunks_.back();
}

}  // namespace runtime
