#pragma once

#include <cstddef>
#include <vector>

#include "runtime/storage.h"
#include "runtime/tensor.h"

namespace runtime {

class MemoryPool {
 public:
  explicit MemoryPool(
      std::size_t default_chunk_size = 1 << 20,
      std::size_t default_alignment = 64);

  Buffer allocate(
      std::size_t bytes,
      std::size_t alignment = alignof(std::max_align_t));

  Tensor allocate_tensor(const Shape& shape, DType dtype = DType::Float32);

  void reset();

  std::size_t used_bytes() const { return used_bytes_; }
  std::size_t peak_bytes() const { return peak_bytes_; }
  std::size_t allocation_count() const { return allocation_count_; }
  std::size_t chunk_count() const { return chunks_.size(); }

 private:
  struct Chunk {
    Buffer backing;
    std::size_t capacity = 0;
    std::size_t offset = 0;
  };

  static std::size_t align_up(std::size_t value, std::size_t alignment);
  Chunk& reserve_chunk(std::size_t min_bytes, std::size_t alignment);

  std::vector<Chunk> chunks_;
  std::size_t default_chunk_size_;
  std::size_t default_alignment_;
  std::size_t used_bytes_ = 0;
  std::size_t peak_bytes_ = 0;
  std::size_t allocation_count_ = 0;
};

}  // namespace runtime
