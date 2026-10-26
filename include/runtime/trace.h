#pragma once

#include <cstddef>
#include <string_view>
#include <chrono>
#include <exception>
#include <string>
#include <utility>

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

class TraceSpan {
 public:
  TraceSpan(TraceSink* sink, std::string_view stage, std::string detail,
            std::string_view event = "stage")
      : sink_(sink), stage_(stage), detail_(std::move(detail)), event_(event),
        exceptions_(std::uncaught_exceptions()), start_(std::chrono::steady_clock::now()) {
    if (sink_) sink_->emit(stage_, event_ + "_start", detail_);
  }
  ~TraceSpan() {
    if (!sink_) return;
    try {
      const double duration = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start_).count();
      sink_->emit(stage_, event_ + (std::uncaught_exceptions() > exceptions_ ? "_error" : "_complete"),
                  detail_, bytes_, duration);
    } catch (...) {
    }
  }
  TraceSpan(const TraceSpan&) = delete;
  TraceSpan& operator=(const TraceSpan&) = delete;
  void output(std::size_t bytes, std::string_view detail = {}) {
    bytes_ = bytes;
    if (!detail.empty()) detail_ += " " + std::string(detail);
  }

 private:
  TraceSink* sink_;
  std::string stage_;
  std::string detail_;
  std::string event_;
  int exceptions_;
  std::chrono::steady_clock::time_point start_;
  std::size_t bytes_ = 0;
};

}  // namespace runtime
