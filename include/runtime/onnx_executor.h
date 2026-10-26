#pragma once

#include <string>
#include <unordered_map>

#include "runtime/backend.h"
#include "runtime/onnx_weights.h"
#include "runtime/onnx_plan.h"

namespace runtime {

struct OnnxStreamState {
  std::unordered_map<std::string, Tensor> recurrent;
  std::unordered_map<std::string, Tensor> padding;
};

class OnnxExecutor {
 public:
  explicit OnnxExecutor(const Backend& backend) : backend_(backend) {}
  void prepare(const OnnxGraph& graph);
  void set_trace_sink(TraceSink* sink, std::size_t node_limit = 1024);

  std::unordered_map<std::string, Tensor> run(
      const OnnxGraph& graph,
      const std::unordered_map<std::string, Tensor>& inputs,
      MemoryPool& pool,
      OnnxStreamState* stream_state = nullptr,
      std::string_view trace_stage = "hush.onnx") const;
  std::unordered_map<std::string, Tensor> run(
      const OnnxGraph& graph,
      const OnnxExecutionPlan& plan,
      const std::unordered_map<std::string, Tensor>& inputs,
      MemoryPool& pool,
      OnnxStreamState* stream_state = nullptr,
      std::string_view trace_stage = "hush.onnx") const;

 private:
  const Backend& backend_;
  TraceSink* trace_ = nullptr;
  mutable std::size_t trace_nodes_remaining_ = 0;
  std::unordered_map<const OnnxGraph*, OnnxExecutionPlan> plans_;
  std::unordered_map<const OnnxGraph*, std::unordered_map<std::string, Tensor>> initializers_;
};

}  // namespace runtime
