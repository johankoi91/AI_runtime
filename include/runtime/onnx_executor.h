#pragma once

#include <string>
#include <unordered_map>

#include "runtime/backend.h"
#include "runtime/onnx_weights.h"
#include "runtime/onnx_plan.h"

namespace runtime {

class OnnxExecutor {
 public:
  explicit OnnxExecutor(const Backend& backend) : backend_(backend) {}

  std::unordered_map<std::string, Tensor> run(
      const OnnxGraph& graph,
      const std::unordered_map<std::string, Tensor>& inputs,
      MemoryPool& pool) const;
  std::unordered_map<std::string, Tensor> run(
      const OnnxGraph& graph,
      const OnnxExecutionPlan& plan,
      const std::unordered_map<std::string, Tensor>& inputs,
      MemoryPool& pool) const;

 private:
  const Backend& backend_;
};

}  // namespace runtime
