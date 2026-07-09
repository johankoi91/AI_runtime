#pragma once

#include <cstddef>
#include <memory>
#include <new>

namespace runtime {

// Buffer 表示“一块原始内存”。
//
// 它本身不关心这块内存里存的是矩阵、向量还是 cache，只负责：
// 1. 记住谁拥有这块内存
// 2. 记住数据起始地址
// 3. 记住总字节数
//
// Tensor 会在 Buffer 之上再补 shape / stride / dtype 这些“语义层”信息。
struct Buffer {
  // owner 负责这块内存的生命周期。
  // 这里用 shared_ptr<void>，是为了让多个 Tensor view 可以共享同一块底层内存。
  std::shared_ptr<void> owner;

  // data 指向真正的原始字节地址。
  std::byte* data = nullptr;

  // bytes 记录整块 buffer 的大小。
  std::size_t bytes = 0;

  explicit operator bool() const { return data != nullptr; }
};

// 创建一块“自己拥有”的新内存。
//
// 这个函数通常用于：
// - Tensor::empty / Tensor::zeros
// - MemoryPool 新建 chunk
//
// alignment 默认为平台常见的最大对齐值，后面做更底层优化时，这里会变得更重要。
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
