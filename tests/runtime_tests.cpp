#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
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
#include "runtime/onnx_weights.h"
#include "runtime/hush_bundle.h"

#include <zlib.h>
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

void append_varint(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
  while (value >= 0x80) {
    bytes.push_back(static_cast<std::uint8_t>(value) | 0x80);
    value >>= 7;
  }
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void append_field(std::vector<std::uint8_t>& message, unsigned field, const std::vector<std::uint8_t>& value) {
  append_varint(message, (static_cast<std::uint64_t>(field) << 3) | 2);
  append_varint(message, value.size());
  message.insert(message.end(), value.begin(), value.end());
}

void test_onnx_weights(TestContext& ctx) {
  std::vector<std::uint8_t> tensor;
  std::vector<std::uint8_t> dimensions;
  append_varint(dimensions, 2);
  append_varint(dimensions, 2);
  append_field(tensor, 1, dimensions);
  append_varint(tensor, (2ull << 3) | 0);
  append_varint(tensor, 1);
  append_field(tensor, 8, {'w', 'e', 'i', 'g', 'h', 't'});
  const float values[] = {1.0f, -2.0f, 3.5f, 4.25f};
  std::vector<std::uint8_t> raw(sizeof(values));
  std::memcpy(raw.data(), values, sizeof(values));
  append_field(tensor, 9, raw);

  std::vector<std::uint8_t> graph;
  append_field(graph, 5, tensor);
  std::vector<std::uint8_t> model;
  append_field(model, 7, graph);
  const std::string path = "/tmp/mini-runtime-stage9-model.onnx";
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
  file.close();

  const auto loaded = runtime::OnnxWeights::read_model(path);
  ctx.expect_true(loaded.size() == 1, "ONNX reader should find one initializer");
  ctx.expect_true(loaded[0].name == "weight", "ONNX reader should preserve initializer name");
  ctx.expect_true(loaded[0].shape == std::vector<std::size_t>({2, 2}), "ONNX reader should preserve shape");
  ctx.expect_close(loaded[0].values[1], -2.0f, 1e-6f, "ONNX reader should decode raw float data");
}

std::vector<std::uint8_t> gzip_bytes(const std::vector<std::uint8_t>& input) {
  z_stream stream{};
  if (deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 16 + MAX_WBITS, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK) {
    throw std::runtime_error("could not initialize test gzip encoder");
  }
  std::vector<std::uint8_t> output(compressBound(input.size()));
  stream.next_in = const_cast<Bytef*>(input.data());
  stream.avail_in = static_cast<uInt>(input.size());
  stream.next_out = output.data();
  stream.avail_out = static_cast<uInt>(output.size());
  if (deflate(&stream, Z_FINISH) != Z_STREAM_END) {
    deflateEnd(&stream);
    throw std::runtime_error("could not encode test gzip");
  }
  output.resize(stream.total_out);
  deflateEnd(&stream);
  return output;
}

void append_tar_entry(std::vector<std::uint8_t>& tar, const std::string& name, const std::string& content) {
  std::vector<std::uint8_t> header(512, 0);
  std::memcpy(header.data(), name.data(), std::min(name.size(), std::size_t(100)));
  std::snprintf(reinterpret_cast<char*>(header.data() + 100), 8, "%07o", 0644);
  std::snprintf(reinterpret_cast<char*>(header.data() + 108), 8, "%07o", 0);
  std::snprintf(reinterpret_cast<char*>(header.data() + 116), 8, "%07o", 0);
  std::snprintf(reinterpret_cast<char*>(header.data() + 124), 12, "%011zo", content.size());
  std::snprintf(reinterpret_cast<char*>(header.data() + 136), 12, "%011o", 0);
  header[156] = '0';
  std::memcpy(header.data() + 257, "ustar\0", 6);
  std::memcpy(header.data() + 263, "00", 2);
  std::fill(header.begin() + 148, header.begin() + 156, ' ');
  unsigned checksum = 0;
  for (std::uint8_t value : header) checksum += value;
  std::snprintf(reinterpret_cast<char*>(header.data() + 148), 8, "%06o ", checksum);
  tar.insert(tar.end(), header.begin(), header.end());
  tar.insert(tar.end(), content.begin(), content.end());
  tar.resize((tar.size() + 511) / 512 * 512, 0);
}

void test_hush_bundle(TestContext& ctx) {
  std::vector<std::uint8_t> tar;
  append_tar_entry(tar, "enc.onnx", "encoder");
  append_tar_entry(tar, "config.ini", "[df]\nsr=16000\n");
  tar.resize(tar.size() + 1024, 0);
  const std::string path = "/tmp/mini-runtime-stage10-model.tar.gz";
  const auto compressed = gzip_bytes(tar);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(compressed.data()), static_cast<std::streamsize>(compressed.size()));
  file.close();
  const runtime::HushModelBundle bundle = runtime::HushModelBundle::read(path);
  ctx.expect_true(bundle.entries().size() == 2, "Hush bundle should list regular files");
  ctx.expect_true(bundle.has("enc.onnx"), "Hush bundle should contain encoder model");
  const auto& config = bundle.entry("config.ini");
  const std::string config_text(reinterpret_cast<const char*>(config.data()), config.size());
  ctx.expect_true(config.size() == 14, "Hush bundle should preserve config length");
  ctx.expect_true(config_text == "[df]\nsr=16000\n", "Hush bundle should preserve config bytes");
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
      {"onnx_weights", test_onnx_weights},
      {"hush_bundle", test_hush_bundle},
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
