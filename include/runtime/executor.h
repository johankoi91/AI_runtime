#pragma once

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
  explicit Executor(const Backend& backend) : backend_(backend) {}

  void run(Graph& graph, MemoryPool& pool) const;

 private:
  const Backend& backend_;
};

}  // namespace runtime
