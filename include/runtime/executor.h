#pragma once

#include "runtime/backend.h"
#include "runtime/graph.h"

namespace runtime {

class Executor {
 public:
  explicit Executor(const Backend& backend) : backend_(backend) {}

  void run(Graph& graph, MemoryPool& pool) const;

 private:
  const Backend& backend_;
};

}  // namespace runtime
