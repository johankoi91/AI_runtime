#pragma once

#include <cstddef>
#include <string_view>

namespace runtime {

class TraceSink {
 public:
  virtual ~TraceSink() = default;

  virtual void emit(
      std::string_view stage,
      std::string_view event,
      std::string_view detail,
      std::size_t bytes = 0,
      double duration_ms = 0.0) = 0;
};

}  // namespace runtime
