#include "runtime/memory_pool.h"

#include <algorithm>
#include <numeric>

namespace runtime {

namespace {

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

  Chunk& chunk = reserve_chunk(bytes, std::max(alignment, default_alignment_));
  const std::size_t aligned_offset = align_up(chunk.offset, alignment);
  const std::size_t consumed = (aligned_offset + bytes) - chunk.offset;
  chunk.offset = aligned_offset + bytes;

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
  const auto bytes = product(shape) * dtype_size(dtype);
  return Tensor(dtype, shape, allocate(bytes));
}

void MemoryPool::reset() {
  for (Chunk& chunk : chunks_) {
    chunk.offset = 0;
  }
  used_bytes_ = 0;
}

std::size_t MemoryPool::align_up(std::size_t value, std::size_t alignment) {
  const std::size_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

MemoryPool::Chunk& MemoryPool::reserve_chunk(
    std::size_t min_bytes,
    std::size_t alignment) {
  if (!chunks_.empty()) {
    Chunk& current = chunks_.back();
    const std::size_t aligned_offset = align_up(current.offset, alignment);
    if ((aligned_offset + min_bytes) <= current.capacity) {
      return current;
    }
  }

  const std::size_t capacity =
      std::max(default_chunk_size_, min_bytes + alignment);

  chunks_.push_back(Chunk{
      .backing = make_owned_buffer(capacity, alignment),
      .capacity = capacity,
      .offset = 0,
  });
  return chunks_.back();
}

}  // namespace runtime
