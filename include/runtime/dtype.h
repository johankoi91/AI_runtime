#pragma once

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace runtime {

enum class DType {
  Float32,
};

inline std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return sizeof(float);
  }
  throw std::runtime_error("unsupported dtype");
}

inline std::string_view dtype_name(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return "float32";
  }
  throw std::runtime_error("unsupported dtype");
}

}  // namespace runtime
