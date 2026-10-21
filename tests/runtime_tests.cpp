#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "runtime/cpu_backend.h"
#include "runtime/audio.h"
#include "runtime/executor.h"
#include "runtime/graph.h"
#include "runtime/kernels.h"
#include "runtime/memory_pool.h"
#include "runtime/tensor.h"
#include "runtime/weights.h"
#include "runtime/spectral.h"

namespace {

bool terminal_supports_color() {
  if (std::getenv("NO_COLOR") != nullptr) {
    return false;
  }
#if defined(_WIN32)
  return false;
#else
  return ::isatty(fileno(stdout)) != 0;
#endif
}

std::string colorize(std::string_view text, std::string_view code, bool enabled) {
  if (!enabled) {
    return std::string(text);
  }
  return "\033[" + std::string(code) + "m" + std::string(text) + "\033[0m";
}

void print_separator(char ch = '=', int width = 56) {
  for (int i = 0; i < width; ++i) {
    std::cout << ch;
  }
  std::cout << "\n";
}

void print_summary_line(std::string_view label, int value) {
  std::cout << std::left << std::setw(18) << label << ": " << value << "\n";
}

// 这是一个非常轻量的测试辅助结构。
// 它不像 gtest 那么正式，但足够当前学习阶段使用。
struct TestContext {
  int checks = 0;
  int failures = 0;

  void expect_true(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
      ++failures;
      std::cerr << "[FAIL] " << message << "\n";
    }
  }

  void expect_close(float actual, float expected, float eps, const std::string& message) {
    ++checks;
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

void test_audio_stream(TestContext& ctx) {
  runtime::AudioStream stream(runtime::StreamConfig{});
  stream.push(runtime::AudioFrame::mono_16k(std::vector<float>(159, 0.25f)));
  ctx.expect_true(!stream.has_frame(), "stream should wait for a complete 10 ms frame");
  stream.push(runtime::AudioFrame::mono_16k(std::vector<float>(1, 0.5f)));
  ctx.expect_true(stream.has_frame(), "stream should expose a complete 10 ms frame");
  runtime::AudioFrame frame = stream.pop();
  ctx.expect_true(frame.sample_count() == 160, "popped frame should contain 160 samples");
  ctx.expect_close(frame.samples().front(), 0.25f, 1e-6f, "first audio sample");
  ctx.expect_close(frame.samples().back(), 0.5f, 1e-6f, "last audio sample");
  ctx.expect_true(!stream.has_frame(), "stream should consume the popped frame");
}

void test_weight_file(TestContext& ctx) {
  const std::string path = "/tmp/mini-runtime-stage1-weights.bin";
  const std::vector<runtime::WeightTensor> expected = {
      {.name = "encoder.bias", .shape = {2}, .values = {0.25f, -0.5f}},
      {.name = "encoder.weight", .shape = {2, 2}, .values = {1.0f, 2.0f, 3.0f, 4.0f}},
  };
  runtime::WeightFile::write(path, expected);
  const std::vector<runtime::WeightTensor> actual = runtime::WeightFile::read(path);
  ctx.expect_true(actual.size() == expected.size(), "weight tensor count should round-trip");
  ctx.expect_true(actual[0].name == expected[0].name, "weight tensor name should round-trip");
  ctx.expect_true(actual[1].shape == expected[1].shape, "weight tensor shape should round-trip");
  ctx.expect_close(actual[1].values[3], 4.0f, 1e-6f, "weight tensor value should round-trip");
}

void test_spectral_round_trip(TestContext& ctx) {
  std::vector<float> samples(640);
  for (std::size_t index = 0; index < samples.size(); ++index) {
    samples[index] = 0.4f * std::sin(static_cast<float>(index) * 0.17f);
  }
  const runtime::Spectrogram spectrogram = runtime::stft(samples, 320, 160);
  ctx.expect_true(spectrogram.bins == 161, "stft should produce 161 positive-frequency bins");
  ctx.expect_true(spectrogram.original_samples == samples.size(), "stft should preserve sample count");
  const std::vector<float> reconstructed = runtime::istft(spectrogram);
  ctx.expect_true(reconstructed.size() == samples.size(), "istft should restore sample count");
  for (std::size_t index = 20; index + 20 < samples.size(); ++index) {
    ctx.expect_close(reconstructed[index], samples[index], 2e-4f, "stft/istft sample");
  }
}

}  // namespace

int main() {
  TestContext ctx;
  const bool use_color = terminal_supports_color();
  const auto suite_start = std::chrono::steady_clock::now();

  struct TestCase {
    std::string name;
    std::function<void(TestContext&)> fn;
  };

  // 当前把测试函数们放进一个数组里，顺序执行。
  const std::vector<TestCase> tests = {
      {"tensor_basics", test_tensor_basics},
      {"add", test_add},
      {"matmul", test_matmul},
      {"memory_pool_reuse", test_memory_pool_reuse},
      {"graph_execution", test_graph_execution},
      {"audio_stream", test_audio_stream},
      {"weight_file", test_weight_file},
      {"spectral_round_trip", test_spectral_round_trip},
  };

  int passed_tests = 0;
  int failed_tests = 0;

  print_separator();
  std::cout << colorize("mini-runtime test runner", "1;36", use_color) << "\n";
  print_separator();
  std::cout << "total test cases: " << tests.size() << "\n\n";

  for (const auto& test : tests) {
    const int failures_before = ctx.failures;
    const int checks_before = ctx.checks;
    const auto start = std::chrono::steady_clock::now();

    std::cout << colorize("[RUN ]", "1;34", use_color) << " " << test.name << "\n";

    try {
      test.fn(ctx);
    } catch (const std::exception& ex) {
      ++ctx.failures;
      std::cerr << "[FAIL] unexpected exception: " << ex.what() << "\n";
    }

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    const int checks_in_test = ctx.checks - checks_before;
    const int failures_in_test = ctx.failures - failures_before;

    if (failures_in_test == 0) {
      ++passed_tests;
      std::cout << colorize("[PASS]", "1;32", use_color) << " " << test.name << " (" << checks_in_test
                << " checks, " << elapsed_ms << " ms)\n\n";
    } else {
      ++failed_tests;
      std::cout << colorize("[FAIL]", "1;31", use_color) << " " << test.name << " (" << failures_in_test
                << " failures, " << checks_in_test << " checks, "
                << elapsed_ms << " ms)\n\n";
    }
  }

  const auto suite_end = std::chrono::steady_clock::now();
  const auto total_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(suite_end - suite_start).count();

  print_separator();
  std::cout << colorize("summary", "1;36", use_color) << "\n";
  print_separator();
  print_summary_line("passed tests", passed_tests);
  print_summary_line("failed tests", failed_tests);
  print_summary_line("total checks", ctx.checks);
  print_summary_line("total time (ms)", static_cast<int>(total_ms));
  print_separator('-');

  if (ctx.failures == 0) {
    std::cout << colorize("All mini runtime tests passed.", "1;32", use_color) << "\n";
    return 0;
  }

  std::cerr << colorize("Some mini runtime checks failed.", "1;31", use_color)
            << " total failing checks: " << ctx.failures << "\n";
  return 1;
}
