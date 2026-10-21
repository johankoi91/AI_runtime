#pragma once

#include <memory>
#include <string>

#include "runtime/trace.h"

namespace runtime {

class InfluxDbSink final : public TraceSink {
 public:
  InfluxDbSink();
  ~InfluxDbSink() override;

  InfluxDbSink(const InfluxDbSink&) = delete;
  InfluxDbSink& operator=(const InfluxDbSink&) = delete;

  void emit(
      std::string_view stage,
      std::string_view event,
      std::string_view detail,
      std::size_t bytes = 0,
      double duration_ms = 0.0) override;

  bool enabled() const;
  const std::string& run_id() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runtime
