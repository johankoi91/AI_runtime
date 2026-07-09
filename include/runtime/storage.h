#pragma once

#include <cstddef>
#include <memory>
#include <new>

namespace runtime {

struct Buffer {
  std::shared_ptr<void> owner;
  std::byte* data = nullptr;
  std::size_t bytes = 0;

  explicit operator bool() const { return data != nullptr; }
};

inline Buffer make_owned_buffer(
    std::size_t bytes,
    std::size_t alignment = alignof(std::max_align_t)) {
  if (bytes == 0) {
    return {};
  }

  void* raw = ::operator new(bytes, std::align_val_t(alignment));
  auto deleter = [alignment](void* ptr) {
    ::operator delete(ptr, std::align_val_t(alignment));
  };

  return {
      .owner = std::shared_ptr<void>(raw, deleter),
      .data = static_cast<std::byte*>(raw),
      .bytes = bytes,
  };
}

}  // namespace runtime
