#pragma once

#include <iosfwd>

#include "runtime/backend.h"
#include "runtime/graph.h"

namespace runtime {

// Executor 的职责非常纯粹：
// 按 Graph 里记录的节点顺序，一个一个执行。
//
// 当前版本里，Executor 不负责优化，也不负责拓扑排序，
// 它只是一个最小可理解的“图执行器”。
class Executor {
 public:
  explicit Executor(
      const Backend& backend,
      std::ostream* logger = nullptr,
      TraceSink* trace = nullptr)
      : backend_(backend), logger_(logger), trace_(trace) {
    backend_.set_trace_sink(trace_);
  }

  void run(Graph& graph, MemoryPool& pool) const;
  void set_logger(std::ostream* logger) { logger_ = logger; }
  void set_trace_sink(TraceSink* trace) {
    trace_ = trace;
    backend_.set_trace_sink(trace_);
  }

 private:
  const Backend& backend_;
  std::ostream* logger_ = nullptr;
  TraceSink* trace_ = nullptr;
};

}  // namespace runtime
