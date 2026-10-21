#include <iomanip>
#include <iostream>
#include <string_view>

#include "runtime/cpu_backend.h"
#include "runtime/executor.h"
#include "runtime/graph.h"
#include "runtime/memory_pool.h"
#include "runtime/tensor.h"

namespace {

std::string_view op_type_name(runtime::OpType op) {
  switch (op) {
    case runtime::OpType::Add:
      return "Add";
    case runtime::OpType::MatMul:
      return "MatMul";
  }
  return "Unknown";
}

void print_section(std::string_view title) {
  std::cout << "\n=== " << title << " ===\n";
}

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

void print_tensor(const std::string& name, const runtime::Tensor& tensor) {
  std::cout << "- " << name << ": " << tensor.debug_string() << "\n";
  std::cout << "  values:\n";

  if (tensor.rank() == 2) {
    print_matrix(tensor);
    return;
  }

  if (tensor.rank() == 1) {
    const float* data = tensor.data<float>();
    std::cout << "  [ ";
    for (std::size_t i = 0; i < tensor.shape()[0]; ++i) {
      std::cout << data[i] << ' ';
    }
    std::cout << "]\n";
    return;
  }

  std::cout << "  (pretty print currently supports rank-1 and rank-2 tensors)\n";
}

void print_graph_plan(const runtime::Graph& graph) {
  const auto& nodes = graph.nodes();
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const runtime::Node& node = nodes[i];
    std::cout << std::setw(2) << (i + 1) << ". " << node.name << " = "
              << op_type_name(node.op) << "(";
    for (std::size_t j = 0; j < node.inputs.size(); ++j) {
      if (j > 0) {
        std::cout << ", ";
      }
      std::cout << node.inputs[j];
    }
    std::cout << ")\n";
  }
}

void print_pool_stats(const runtime::MemoryPool& pool) {
  std::cout << "- used bytes: " << pool.used_bytes() << "\n";
  std::cout << "- peak bytes: " << pool.peak_bytes() << "\n";
  std::cout << "- allocation count: " << pool.allocation_count() << "\n";
  std::cout << "- chunk count: " << pool.chunk_count() << "\n";
}

float matrix_at(const runtime::Tensor& tensor, std::size_t row, std::size_t col) {
  return tensor.data<float>()[row * tensor.shape()[1] + col];
}

void print_matmul_hint(const runtime::Tensor& lhs, const runtime::Tensor& rhs) {
  const std::size_t m = lhs.shape()[0];
  const std::size_t k = lhs.shape()[1];
  const std::size_t n = rhs.shape()[1];

  std::cout << "We compute hidden = a @ b.\n";
  std::cout << "For each output cell hidden[i, j], we multiply row i of a with column j of b.\n";

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      std::cout << "  hidden[" << i << ", " << j << "] = ";
      float sum = 0.0f;
      for (std::size_t p = 0; p < k; ++p) {
        const float left = matrix_at(lhs, i, p);
        const float right = matrix_at(rhs, p, j);
        sum += left * right;
        if (p > 0) {
          std::cout << " + ";
        }
        std::cout << left << "*" << right;
      }
      std::cout << " = " << sum << "\n";
    }
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
  runtime::Executor executor(backend, &std::cout);

  print_section("Mini Runtime Demo");
  std::cout << "Backend: " << backend.name() << "\n";

  print_section("Input Tensors");
  print_tensor("a", graph.tensor("a"));
  print_tensor("b", graph.tensor("b"));
  print_tensor("bias", graph.tensor("bias"));

  print_section("Execution Plan");
  print_graph_plan(graph);

  print_section("MatMul Hand Calculation Hint");
  print_matmul_hint(graph.tensor("a"), graph.tensor("b"));

  print_section("Memory Pool Before Run");
  print_pool_stats(pool);

  print_section("Run Graph");
  std::cout << "Running executor over " << graph.nodes().size() << " nodes...\n";
  executor.run(graph, pool);

  const runtime::Tensor& hidden = graph.tensor("hidden");
  const runtime::Tensor& out = graph.tensor("out");

  print_section("Intermediate Tensor");
  print_tensor("hidden", hidden);

  print_section("Final Output");
  print_tensor("out", out);

  print_section("Memory Pool After Run");
  print_pool_stats(pool);

  return 0;
}
