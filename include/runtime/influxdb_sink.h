#pragma once

#include <memory>
#include <string>
#include <chrono>

#include "runtime/trace.h"

namespace runtime {

struct InfluxDbConfig {
  std::string url;
  std::string org;
  std::string bucket;
  std::string token;
  std::string run_id;
  std::size_t max_pending_bytes = 2 * 1024 * 1024;
  bool enabled = true;
};

struct InfluxDbStatus {
  std::size_t accepted_events = 0;
  std::size_t uploaded_events = 0;
  std::size_t dropped_events = 0;
  std::size_t pending_bytes = 0;
  std::string error;
};

class InfluxDbSink final : public TraceSink {
 public:
  explicit InfluxDbSink(InfluxDbConfig config = {});
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
  InfluxDbStatus status() const;
  bool flush(std::chrono::milliseconds timeout = std::chrono::seconds(10));

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runtime
