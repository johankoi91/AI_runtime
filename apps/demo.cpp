#include <iomanip>
#include <iostream>

#include "runtime/cpu_backend.h"
#include "runtime/executor.h"
#include "runtime/graph.h"
#include "runtime/memory_pool.h"
#include "runtime/tensor.h"

namespace {

// 这个函数只是为了让 demo 输出更适合肉眼阅读。
void print_matrix(const runtime::Tensor& tensor) {
  const auto& shape = tensor.shape();
  const float* data = tensor.data<float>();

  for (std::size_t i = 0; i < shape[0]; ++i) {
    std::cout << "[ ";
    for (std::size_t j = 0; j < shape[1]; ++j) {
      std::cout << std::setw(6) << data[i * shape[1] + j] << ' ';
    }
    std::cout << "]\n";
  }
}

}  // namespace

int main() {
  // 这个 demo 演示的是一条非常小的计算图：
  //
  // hidden = a @ b
  // out = hidden + bias
  //
  // 它能帮助你把 Tensor、Graph、Executor、Backend 这些概念一次串起来。
  runtime::Graph graph;
  graph.set_input(
      "a",
      runtime::Tensor::from_vector({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  graph.set_input(
      "b",
      runtime::Tensor::from_vector({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  graph.set_input(
      "bias",
      runtime::Tensor::from_vector({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f}));

  graph.add_node(runtime::Node{
      .name = "hidden",
      .op = runtime::OpType::MatMul,
      .inputs = {"a", "b"},
  });
  graph.add_node(runtime::Node{
      .name = "out",
      .op = runtime::OpType::Add,
      .inputs = {"hidden", "bias"},
  });

  // pool 负责中间结果的内存分配。
  runtime::MemoryPool pool(4 * 1024);

  // backend 决定“用哪个后端来执行图里的 op”。
  runtime::CPUBackend backend;
  runtime::Executor executor(backend);
  executor.run(graph, pool);

  const runtime::Tensor& out = graph.tensor("out");

  // debug_string 会把 dtype / shape / strides / contiguous 状态打印出来。
  std::cout << "Backend: " << backend.name() << "\n";
  std::cout << "Output tensor: " << out.debug_string() << "\n";
  std::cout << "Values:\n";
  print_matrix(out);
  std::cout << "Pool used bytes: " << pool.used_bytes() << "\n";
  std::cout << "Pool peak bytes: " << pool.peak_bytes() << "\n";
  std::cout << "Pool chunks: " << pool.chunk_count() << "\n";

  return 0;
}
