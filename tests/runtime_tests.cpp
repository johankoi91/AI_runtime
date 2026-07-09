#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "runtime/cpu_backend.h"
#include "runtime/executor.h"
#include "runtime/graph.h"
#include "runtime/kernels.h"
#include "runtime/memory_pool.h"
#include "runtime/tensor.h"

namespace {

// 这是一个非常轻量的测试辅助结构。
// 它不像 gtest 那么正式，但足够当前学习阶段使用。
struct TestContext {
  int failures = 0;

  void expect_true(bool condition, const std::string& message) {
    if (!condition) {
      ++failures;
      std::cerr << "[FAIL] " << message << "\n";
    }
  }

  void expect_close(float actual, float expected, float eps, const std::string& message) {
    if (std::fabs(actual - expected) > eps) {
      ++failures;
      std::cerr << "[FAIL] " << message << " expected=" << expected
                << " actual=" << actual << "\n";
    }
  }
};

void test_tensor_basics(TestContext& ctx) {
  // 这个测试主要验证 Tensor 最基本的元信息是否正确。
  runtime::Tensor tensor = runtime::Tensor::zeros({2, 3});
  ctx.expect_true(tensor.rank() == 2, "tensor rank should be 2");
  ctx.expect_true(tensor.numel() == 6, "tensor numel should be 6");
  ctx.expect_true(tensor.is_contiguous(), "tensor should be contiguous");
  ctx.expect_true(
      tensor.strides() == runtime::Strides({3, 1}),
      "tensor strides should be [3, 1]");
}

void test_add(TestContext& ctx) {
  // 逐元素加法是最简单的 kernel，适合作为第一个数值正确性测试。
  runtime::Tensor lhs = runtime::Tensor::from_vector({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  runtime::Tensor rhs = runtime::Tensor::from_vector({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

  runtime::MemoryPool pool;
  runtime::Tensor out = runtime::cpu::add(lhs, rhs, &pool);
  const float* data = out.data<float>();

  ctx.expect_close(data[0], 6.0f, 1e-5f, "add[0]");
  ctx.expect_close(data[3], 12.0f, 1e-5f, "add[3]");
}

void test_matmul(TestContext& ctx) {
  // 这里验证的是最小矩阵乘法路径是否正确。
  runtime::Tensor lhs = runtime::Tensor::from_vector({2, 3}, {1, 2, 3, 4, 5, 6});
  runtime::Tensor rhs = runtime::Tensor::from_vector({3, 2}, {1, 2, 3, 4, 5, 6});

  runtime::MemoryPool pool;
  runtime::Tensor out = runtime::cpu::matmul(lhs, rhs, &pool);
  const float* data = out.data<float>();

  ctx.expect_close(data[0], 22.0f, 1e-5f, "matmul[0,0]");
  ctx.expect_close(data[1], 28.0f, 1e-5f, "matmul[0,1]");
  ctx.expect_close(data[2], 49.0f, 1e-5f, "matmul[1,0]");
  ctx.expect_close(data[3], 64.0f, 1e-5f, "matmul[1,1]");
}

void test_memory_pool_reuse(TestContext& ctx) {
  // 这个测试不关心数值，而是关心 reset 之后是否真的复用了已有 chunk。
  runtime::MemoryPool pool(256);
  runtime::Tensor first = pool.allocate_tensor({4, 4});
  runtime::Tensor second = pool.allocate_tensor({4, 4});
  (void)first;
  (void)second;

  const std::size_t chunk_count_before_reset = pool.chunk_count();
  ctx.expect_true(chunk_count_before_reset >= 1, "pool should have at least one chunk");
  ctx.expect_true(pool.used_bytes() > 0, "pool should track used bytes");

  pool.reset();
  ctx.expect_true(pool.used_bytes() == 0, "pool reset should clear used bytes");

  runtime::Tensor third = pool.allocate_tensor({4, 4});
  (void)third;
  ctx.expect_true(
      pool.chunk_count() == chunk_count_before_reset,
      "pool should reuse existing chunks after reset");
}

void test_graph_execution(TestContext& ctx) {
  // 这个测试把前面的模块串起来：
  // Tensor -> Graph -> Backend -> Executor -> 输出结果
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

  runtime::MemoryPool pool;
  runtime::CPUBackend backend;
  runtime::Executor executor(backend);
  executor.run(graph, pool);

  const runtime::Tensor& out = graph.tensor("out");
  const float* data = out.data<float>();

  ctx.expect_close(data[0], 32.0f, 1e-5f, "graph out[0,0]");
  ctx.expect_close(data[1], 48.0f, 1e-5f, "graph out[0,1]");
  ctx.expect_close(data[2], 79.0f, 1e-5f, "graph out[1,0]");
  ctx.expect_close(data[3], 104.0f, 1e-5f, "graph out[1,1]");
}

}  // namespace

int main() {
  TestContext ctx;

  // 当前把测试函数们放进一个数组里，顺序执行。
  const std::vector<std::function<void(TestContext&)>> tests = {
      test_tensor_basics,
      test_add,
      test_matmul,
      test_memory_pool_reuse,
      test_graph_execution,
  };

  for (const auto& test : tests) {
    try {
      test(ctx);
    } catch (const std::exception& ex) {
      ++ctx.failures;
      std::cerr << "[FAIL] unexpected exception: " << ex.what() << "\n";
    }
  }

  if (ctx.failures == 0) {
    std::cout << "All mini runtime tests passed.\n";
    return 0;
  }

  std::cerr << ctx.failures << " test checks failed.\n";
  return 1;
}
