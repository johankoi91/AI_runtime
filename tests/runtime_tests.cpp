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
#include "runtime/hush_session.h"
#include "runtime/hush_frontend.h"
#include "runtime/hush_model.h"
#include "runtime/onnx_executor.h"
#include "runtime/onnx_plan.h"

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

void test_mul(TestContext& ctx) {
  runtime::CPUBackend backend;
  runtime::MemoryPool pool;
  const auto output = backend.mul(
      runtime::Tensor::from_vector({2}, {2.0f, -3.0f}),
      runtime::Tensor::from_vector({2}, {4.0f, -2.0f}), pool);
  ctx.expect_close(output.data<float>()[0], 8.0f, 1e-6f, "mul[0]");
  ctx.expect_close(output.data<float>()[1], 6.0f, 1e-6f, "mul[1]");
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

void append_string_field(std::vector<std::uint8_t>& message, unsigned field, const std::string& value) {
  append_field(message, field, std::vector<std::uint8_t>(value.begin(), value.end()));
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
  std::vector<std::uint8_t> node;
  append_field(node, 3, {'a', 'd', 'd'});
  append_field(node, 4, {'A', 'd', 'd'});
  append_field(node, 1, {'x'});
  append_field(node, 1, {'y'});
  append_field(node, 2, {'z'});
  std::vector<std::uint8_t> stride_attribute;
  append_string_field(stride_attribute, 1, "strides");
  std::vector<std::uint8_t> stride_values;
  append_varint(stride_values, 2);
  append_field(stride_attribute, 8, stride_values);
  append_field(node, 5, stride_attribute);
  append_field(graph, 1, node);
  append_field(graph, 2, {'g', 'r', 'a', 'p', 'h'});
  std::vector<std::uint8_t> graph_input;
  append_string_field(graph_input, 1, "x");
  std::vector<std::uint8_t> graph_output;
  append_string_field(graph_output, 1, "z");
  append_field(graph, 11, graph_input);
  append_field(graph, 12, graph_output);
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
  const runtime::OnnxGraph graph_info = runtime::OnnxWeights::read_graph(path);
  ctx.expect_true(graph_info.name == "graph", "ONNX reader should preserve graph name");
  ctx.expect_true(graph_info.inputs == std::vector<std::string>({"x"}), "ONNX reader should preserve graph inputs");
  ctx.expect_true(graph_info.outputs == std::vector<std::string>({"z"}), "ONNX reader should preserve graph outputs");
  ctx.expect_true(graph_info.nodes.size() == 1, "ONNX reader should preserve node count");
  ctx.expect_true(graph_info.nodes[0].op_type == "Add", "ONNX reader should preserve node op type");
  ctx.expect_true(graph_info.nodes[0].inputs.size() == 2 && graph_info.nodes[0].outputs[0] == "z",
                  "ONNX reader should preserve node inputs and outputs");
  const auto& stride = graph_info.attribute(graph_info.nodes[0], "strides");
  ctx.expect_true(stride.integers.size() == 1 && stride.integers[0] == 2,
                  "ONNX reader should preserve repeated integer attributes");
  ctx.expect_close(graph_info.initializer("weight").values[2], 3.5f, 1e-6f,
                   "ONNX graph should map initializer by name");

  runtime::CPUBackend cpu_backend;
  runtime::OnnxExecutor onnx_executor(cpu_backend);
  runtime::MemoryPool pool;
  const auto executed = onnx_executor.run(
      graph_info,
      {{"x", runtime::Tensor::from_vector({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f})},
       {"y", runtime::Tensor::from_vector({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f})}},
      pool);
  ctx.expect_close(executed.at("z").data<float>()[0], 6.0f, 1e-6f,
                   "ONNX executor should map Add to backend");
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

std::vector<std::uint8_t> make_minimal_onnx(const std::string& op_type) {
  std::vector<std::uint8_t> node;
  append_field(node, 1, {'x'});
  append_field(node, 2, {'y'});
  append_string_field(node, 3, "node");
  append_string_field(node, 4, op_type);
  std::vector<std::uint8_t> graph;
  append_field(graph, 1, node);
  append_field(graph, 2, {'g'});
  std::vector<std::uint8_t> input;
  append_string_field(input, 1, "x");
  std::vector<std::uint8_t> output;
  append_string_field(output, 1, "y");
  append_field(graph, 11, input);
  append_field(graph, 12, output);
  std::vector<std::uint8_t> graph_input;
  append_string_field(graph_input, 1, "x");
  std::vector<std::uint8_t> graph_output;
  append_string_field(graph_output, 1, "y");
  append_field(graph, 3, graph_input);
  append_field(graph, 6, graph_output);
  std::vector<std::uint8_t> model;
  append_field(model, 7, graph);
  return model;
}

void test_hush_model(TestContext& ctx) {
  std::vector<std::uint8_t> tar;
  const auto encoder_onnx = make_minimal_onnx("Conv");
  const auto erb_onnx = make_minimal_onnx("Relu");
  const auto df_onnx = make_minimal_onnx("GRU");
  append_tar_entry(tar, "config.ini", "[df]\nsr=16000\n");
  append_tar_entry(tar, "enc.onnx", std::string(encoder_onnx.begin(), encoder_onnx.end()));
  append_tar_entry(tar, "erb_dec.onnx", std::string(erb_onnx.begin(), erb_onnx.end()));
  append_tar_entry(tar, "df_dec.onnx", std::string(df_onnx.begin(), df_onnx.end()));
  tar.resize(tar.size() + 1024, 0);
  const std::string path = "/tmp/mini-runtime-stage12-model.tar.gz";
  const auto compressed = gzip_bytes(tar);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(compressed.data()), static_cast<std::streamsize>(compressed.size()));
  file.close();
  const runtime::HushModel model = runtime::HushModel::load(path);
  ctx.expect_true(model.config_ini.find("sr=16000") != std::string::npos, "Hush model should load config");
  ctx.expect_true(model.config.sample_rate == 16000 && model.config.hop_size == 160,
                  "Hush model should parse audio configuration");
  ctx.expect_true(model.encoder.nodes[0].op_type == "Conv", "Hush model should load encoder graph");
  ctx.expect_true(model.erb_decoder.nodes[0].op_type == "Relu", "Hush model should load ERB decoder graph");
  ctx.expect_true(model.df_decoder.nodes[0].op_type == "GRU", "Hush model should load DF decoder graph");
  const runtime::HushGraphReport report = runtime::inspect_hush_graph(model.encoder);
  ctx.expect_true(report.node_count == 1, "Hush graph report should count nodes");
  ctx.expect_true(report.unsupported_ops.empty(), "minimal Hush graph should use supported ops");
}

void test_onnx_conv_attributes(TestContext& ctx) {
  runtime::OnnxGraph graph;
  runtime::OnnxNode node;
  node.name = "conv";
  node.op_type = "Conv";
  node.inputs = {"x", "w", "b"};
  node.outputs = {"y"};
  node.attributes.push_back({"strides", 0, {1, 1}, {}});
  node.attributes.push_back({"pads", 0, {1, 1, 1, 1}, {}});
  node.attributes.push_back({"dilations", 0, {1, 1}, {}});
  node.attributes.push_back({"group", 1, {}, {}});
  graph.nodes.push_back(node);
  graph.initializers = {
      {.name = "w", .shape = {1, 1, 2, 2}, .values = {1.0f, 0.0f, 0.0f, 1.0f}},
      {.name = "b", .shape = {1}, .values = {0.0f}},
  };
  runtime::CPUBackend backend;
  runtime::OnnxExecutor executor(backend);
  runtime::MemoryPool pool;
  const auto outputs = executor.run(
      graph, {{"x", runtime::Tensor::from_vector({1, 1, 2, 2}, {1, 2, 3, 4})}}, pool);
  ctx.expect_true(outputs.at("y").shape() == runtime::Shape({1, 1, 3, 3}),
                  "ONNX Conv attributes should map to padded 2D Conv");
}

void test_onnx_conv_transpose(TestContext& ctx) {
  runtime::OnnxGraph graph;
  runtime::OnnxNode node;
  node.name = "deconv";
  node.op_type = "ConvTranspose";
  node.inputs = {"x", "w", "b"};
  node.outputs = {"y"};
  node.attributes.push_back({"strides", 0, {2, 2}, {}});
  node.attributes.push_back({"pads", 0, {0, 0, 0, 0}, {}});
  graph.nodes.push_back(node);
  graph.initializers = {
      {.name = "w", .shape = {1, 1, 2, 2}, .values = {1.0f, 0.5f, -1.0f, 2.0f}},
      {.name = "b", .shape = {1}, .values = {0.25f}},
  };
  runtime::CPUBackend backend;
  runtime::OnnxExecutor executor(backend);
  runtime::MemoryPool pool;
  const auto outputs = executor.run(
      graph, {{"x", runtime::Tensor::from_vector({1, 1, 2, 2}, {1, 2, 3, 4})}}, pool);
  ctx.expect_true(outputs.at("y").shape() == runtime::Shape({1, 1, 4, 4}),
                  "ONNX ConvTranspose should map to backend");
  ctx.expect_close(outputs.at("y").data<float>()[0], 1.25f, 1e-6f,
                   "ONNX ConvTranspose should preserve bias and kernel output");
}

void test_onnx_shape_constant(TestContext& ctx) {
  runtime::OnnxGraph graph;
  runtime::OnnxNode shape;
  shape.name = "shape";
  shape.op_type = "Shape";
  shape.inputs = {"x"};
  shape.outputs = {"shape_out"};
  graph.nodes.push_back(shape);
  runtime::OnnxNode constant;
  constant.name = "constant";
  constant.op_type = "Constant";
  constant.outputs = {"constant_out"};
  runtime::WeightTensor value{.name = "", .shape = {2}, .values = {7.0f, 9.0f}};
  constant.attributes.push_back({.name = "value", .has_tensor = true, .tensor = value});
  graph.nodes.push_back(constant);
  runtime::CPUBackend backend;
  runtime::OnnxExecutor executor(backend);
  runtime::MemoryPool pool;
  const auto outputs = executor.run(
      graph, {{"x", runtime::Tensor::from_vector({3, 4}, std::vector<float>(12, 1.0f))}}, pool);
  ctx.expect_true(outputs.at("shape_out").shape() == runtime::Shape({2}),
                  "Shape should return one value per input dimension");
  ctx.expect_close(outputs.at("shape_out").data<float>()[1], 4.0f, 1e-6f,
                   "Shape should return input dimensions");
  ctx.expect_close(outputs.at("constant_out").data<float>()[1], 9.0f, 1e-6f,
                   "Constant should return tensor attribute values");
}

void test_onnx_execution_plan(TestContext& ctx) {
  runtime::OnnxGraph graph;
  graph.nodes = {
      {.name = "second", .op_type = "Relu", .inputs = {"a"}, .outputs = {"b"}},
      {.name = "first", .op_type = "Relu", .inputs = {"x"}, .outputs = {"a"}},
  };
  const runtime::OnnxExecutionPlan plan = runtime::OnnxExecutionPlan::compile(graph);
  ctx.expect_true(plan.nodes().size() == 2, "execution plan should contain all nodes");
  ctx.expect_true(plan.nodes()[0].graph_index == 1 && plan.nodes()[1].graph_index == 0,
                  "execution plan should topologically order nodes");
  ctx.expect_true(plan.output_users("a") == 1, "execution plan should count value users");
  ctx.expect_true(plan.peak_live_values() > 0, "execution plan should report live values");
}

void test_onnx_connection_ops(TestContext& ctx) {
  runtime::OnnxGraph graph;
  graph.initializers = {
      {.name = "axes", .shape = {1}, .values = {1.0f}},
      {.name = "indices", .shape = {2}, .values = {1.0f, 0.0f}},
      {.name = "pads", .shape = {4}, .values = {1.0f, 1.0f, 1.0f, 1.0f}},
  };
  graph.nodes = {
      {.name = "unsqueeze", .op_type = "Unsqueeze", .inputs = {"x", "axes"}, .outputs = {"u"}},
      {.name = "squeeze", .op_type = "Squeeze", .inputs = {"u", "axes"}, .outputs = {"s"}},
      {.name = "gather", .op_type = "Gather", .inputs = {"s", "indices"}, .outputs = {"g"}},
      {.name = "pad", .op_type = "Pad", .inputs = {"g", "pads"}, .outputs = {"p"}},
  };
  runtime::CPUBackend backend;
  runtime::OnnxExecutor executor(backend);
  runtime::MemoryPool pool;
  const auto outputs = executor.run(
      graph, {{"x", runtime::Tensor::from_vector({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f})}}, pool);
  ctx.expect_true(outputs.at("u").shape() == runtime::Shape({2, 1, 2}), "Unsqueeze should add a unit dimension");
  ctx.expect_true(outputs.at("s").shape() == runtime::Shape({2, 2}), "Squeeze should remove a unit dimension");
  ctx.expect_close(outputs.at("g").data<float>()[0], 3.0f, 1e-6f, "Gather should reorder rows");
  ctx.expect_true(outputs.at("p").shape() == runtime::Shape({4, 4}), "Pad should expand both dimensions");
}

void test_onnx_slice(TestContext& ctx) {
  runtime::OnnxGraph graph;
  graph.initializers = {
      {.name = "starts", .shape = {1}, .values = {1.0f}},
      {.name = "ends", .shape = {1}, .values = {3.0f}},
      {.name = "axes", .shape = {1}, .values = {0.0f}},
      {.name = "steps", .shape = {1}, .values = {1.0f}},
  };
  graph.nodes.push_back({.name = "slice", .op_type = "Slice",
                         .inputs = {"x", "starts", "ends", "axes", "steps"}, .outputs = {"y"}});
  runtime::CPUBackend backend;
  runtime::OnnxExecutor executor(backend);
  runtime::MemoryPool pool;
  const auto outputs = executor.run(
      graph, {{"x", runtime::Tensor::from_vector({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8})}}, pool);
  ctx.expect_true(outputs.at("y").shape() == runtime::Shape({2, 2}), "Slice should reduce the selected axis");
  ctx.expect_close(outputs.at("y").data<float>()[0], 3.0f, 1e-6f, "Slice should start at requested index");
}

void test_onnx_einsum(TestContext& ctx) {
  runtime::OnnxGraph graph;
  runtime::OnnxNode node;
  node.name = "einsum";
  node.op_type = "Einsum";
  node.inputs = {"a", "b"};
  node.outputs = {"y"};
  node.attributes.push_back({.name = "equation", .string = "ij,jk->ik"});
  graph.nodes.push_back(node);
  runtime::CPUBackend backend;
  runtime::OnnxExecutor executor(backend);
  runtime::MemoryPool pool;
  const auto outputs = executor.run(
      graph, {{"a", runtime::Tensor::from_vector({1, 2}, {2.0f, 3.0f})},
              {"b", runtime::Tensor::from_vector({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f})}}, pool);
  ctx.expect_close(outputs.at("y").data<float>()[0], 11.0f, 1e-6f,
                   "Einsum matrix equation should use matmul");
}

void test_shape_kernels(TestContext& ctx) {
  const runtime::Tensor input = runtime::Tensor::from_vector(
      {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  const runtime::Tensor reshaped = runtime::cpu::reshape(input, {3, 2});
  ctx.expect_true(reshaped.shape() == runtime::Shape({3, 2}), "reshape should update shape");
  ctx.expect_close(reshaped.data<float>()[4], 5.0f, 1e-6f, "reshape should preserve values");

  const runtime::Tensor transposed = runtime::cpu::transpose(input, {1, 0});
  ctx.expect_true(transposed.shape() == runtime::Shape({3, 2}), "transpose should permute shape");
  ctx.expect_close(transposed.data<float>()[1], 4.0f, 1e-6f, "transpose should permute values");

  const runtime::Tensor lhs = runtime::Tensor::from_vector({2, 1}, {1.0f, 2.0f});
  const runtime::Tensor rhs = runtime::Tensor::from_vector({2, 2}, {3.0f, 4.0f, 5.0f, 6.0f});
  const std::vector<const runtime::Tensor*> parts = {&lhs, &rhs};
  const runtime::Tensor joined = runtime::cpu::concat(parts, 1);
  ctx.expect_true(joined.shape() == runtime::Shape({2, 3}), "concat should expand axis");
  ctx.expect_close(joined.data<float>()[3], 2.0f, 1e-6f, "concat should preserve row order");
  ctx.expect_close(joined.data<float>()[5], 6.0f, 1e-6f, "concat should copy final value");
}

void test_hush_stream_session(TestContext& ctx) {
  runtime::HushModel model;
  model.config_ini = "[df]\nsr=16000\nfft_size=320\nhop_size=160\nnb_erb=32\nnb_df=64\ndf_order=5\n";
  model.config = runtime::HushConfig::from_ini(model.config_ini);
  runtime::OnnxNode node;
  node.name = "state_add";
  node.op_type = "Add";
  node.inputs = {"x", "h"};
  node.outputs = {"y"};
  model.encoder.nodes.push_back(node);
  model.encoder.inputs = {"x", "h"};
  model.encoder.outputs = {"y"};
  runtime::CPUBackend backend;
  runtime::HushStreamSession session(model, backend, {{"h", "y"}});
  runtime::MemoryPool pool;

  const auto first = session.process_encoder(
      {{"x", runtime::Tensor::from_vector({2}, {1.0f, 2.0f})},
       {"h", runtime::Tensor::from_vector({2}, {10.0f, 20.0f})}},
      pool);
  ctx.expect_close(first.at("y").data<float>()[0], 11.0f, 1e-6f, "session first state output");
  ctx.expect_true(session.frame_index() == 1, "session should count first frame");
  ctx.expect_true(session.has_state("h"), "session should retain encoder state");

  bool missing_input_rejected = false;
  try {
    session.process_encoder({{"h", runtime::Tensor::from_vector({2}, {0.0f, 0.0f})}}, pool);
  } catch (const std::runtime_error&) {
    missing_input_rejected = true;
  }
  ctx.expect_true(missing_input_rejected, "session should reject missing encoder input");

  bool invalid_frame_rejected = false;
  try {
    session.process_frame(runtime::AudioFrame::mono_16k(std::vector<float>(80)), {}, pool);
  } catch (const std::invalid_argument&) {
    invalid_frame_rejected = true;
  }
  ctx.expect_true(invalid_frame_rejected, "session should reject a frame with wrong hop size");

  runtime::HushModel feature_model;
  feature_model.config = runtime::HushConfig{};
  runtime::OnnxNode feature_node;
  feature_node.name = "feature_add";
  feature_node.op_type = "Add";
  feature_node.inputs = {"erb", "h"};
  feature_node.outputs = {"y"};
  feature_model.encoder.nodes = {feature_node};
  feature_model.encoder.inputs = {"erb", "h"};
  feature_model.encoder.outputs = {"y"};
  runtime::HushStreamSession feature_session(feature_model, backend, {{"h", "y"}});
  runtime::MemoryPool feature_pool;
  feature_session.process_encoder(
      {{"erb", runtime::Tensor::zeros({1, 32})},
       {"h", runtime::Tensor::zeros({1, 32})}}, feature_pool);
  feature_pool.reset();
  const auto feature_output = feature_session.process_audio_frame(
      runtime::AudioFrame::mono_16k(std::vector<float>(160, 0.1f)), "erb", feature_pool);
  ctx.expect_true(feature_output.at("y").shape() == runtime::Shape({1, 32}),
                  "audio frame should feed ERB feature into encoder");

  runtime::HushModel dual_feature_model;
  dual_feature_model.config = runtime::HushConfig{};
  runtime::OnnxNode dual_node;
  dual_node.name = "dual_add";
  dual_node.op_type = "Relu";
  dual_node.inputs = {"erb"};
  dual_node.outputs = {"y"};
  dual_feature_model.encoder.nodes = {dual_node};
  dual_feature_model.encoder.inputs = {"erb", "df"};
  dual_feature_model.encoder.outputs = {"y"};
  runtime::HushStreamSession dual_session(dual_feature_model, backend);
  runtime::MemoryPool dual_pool;
  const auto dual_output = dual_session.process_audio_frame(
      runtime::AudioFrame::mono_16k(std::vector<float>(160, 0.1f)), "erb", "df", dual_pool);
  ctx.expect_true(dual_output.at("y").shape() == runtime::Shape({1, 32}),
                  "audio frame should feed ERB and DF features into encoder");

  runtime::HushModel decoder_model;
  decoder_model.config = runtime::HushConfig{};
  runtime::OnnxNode encoder_node;
  encoder_node.name = "encoder_relu";
  encoder_node.op_type = "Relu";
  encoder_node.inputs = {"input"};
  encoder_node.outputs = {"hidden"};
  decoder_model.encoder.nodes = {encoder_node};
  decoder_model.encoder.inputs = {"input"};
  decoder_model.encoder.outputs = {"hidden"};
  runtime::OnnxNode erb_node;
  erb_node.name = "erb_relu";
  erb_node.op_type = "Relu";
  erb_node.inputs = {"hidden"};
  erb_node.outputs = {"erb_out"};
  decoder_model.erb_decoder.nodes = {erb_node};
  decoder_model.erb_decoder.inputs = {"hidden"};
  decoder_model.erb_decoder.outputs = {"erb_out"};
  runtime::OnnxNode df_node = erb_node;
  df_node.name = "df_tanh";
  df_node.op_type = "Tanh";
  df_node.outputs = {"df_out"};
  decoder_model.df_decoder.nodes = {df_node};
  decoder_model.df_decoder.inputs = {"hidden"};
  decoder_model.df_decoder.outputs = {"df_out"};
  runtime::HushStreamSession decoder_session(decoder_model, backend);
  runtime::MemoryPool decoder_pool;
  const auto decoder_outputs = decoder_session.process_decoders(
      {{"hidden", runtime::Tensor::from_vector({2}, {-1.0f, 0.5f})}}, {}, {}, decoder_pool);
  ctx.expect_close(decoder_outputs.erb.at("erb_out").data<float>()[0], 0.0f, 1e-6f,
                   "ERB decoder should receive encoder output");
  ctx.expect_close(decoder_outputs.df.at("df_out").data<float>()[1], std::tanh(0.5f), 1e-6f,
                   "DF decoder should receive encoder output");

  runtime::HushModel full_model = decoder_model;
  full_model.encoder.inputs = {"erb", "df"};
  full_model.encoder.nodes[0].inputs = {"erb"};
  full_model.encoder.nodes[0].outputs = {"hidden"};
  full_model.encoder.outputs = {"hidden"};
  runtime::HushStreamSession full_session(full_model, backend);
  runtime::MemoryPool full_pool;
  const runtime::AudioFrame enhanced = full_session.process_audio(
      runtime::AudioFrame::mono_16k(std::vector<float>(160, 0.1f)),
      "erb", "df", "erb_out", backend, full_pool);
  ctx.expect_true(enhanced.sample_count() == 160, "full Hush audio pipeline should preserve frame length");
  ctx.expect_true(std::isfinite(enhanced.samples()[80]), "full Hush audio output should be finite");

  pool.reset();
  const auto second = session.process_encoder(
      {{"x", runtime::Tensor::from_vector({2}, {2.0f, 3.0f})}}, pool);
  ctx.expect_close(second.at("y").data<float>()[0], 13.0f, 1e-6f, "session second state output");
  ctx.expect_true(session.frame_index() == 2, "session should count second frame");

  session.reset();
  ctx.expect_true(session.frame_index() == 0, "session reset should clear frame count");
  ctx.expect_true(!session.has_state("h"), "session reset should clear state");

  pool.reset();
  const auto frame_output = session.process_frame(
      runtime::AudioFrame::mono_16k(std::vector<float>(160, 0.0f)),
      {{"x", runtime::Tensor::from_vector({2}, {4.0f, 5.0f})},
       {"h", runtime::Tensor::from_vector({2}, {1.0f, 1.0f})}},
      pool);
  ctx.expect_close(frame_output.at("y").data<float>()[0], 5.0f, 1e-6f,
                   "session audio frame entry should execute encoder");

  bool rejected_frame = false;
  try {
    session.process_frame(runtime::AudioFrame(runtime::AudioFormat{8'000, 1}, std::vector<float>(80)),
                          {{"x", runtime::Tensor::from_vector({2}, {1.0f, 1.0f})},
                           {"h", runtime::Tensor::from_vector({2}, {1.0f, 1.0f})}},
                          pool);
  } catch (const std::invalid_argument&) {
    rejected_frame = true;
  }
  ctx.expect_true(rejected_frame, "session should reject non-Hush audio format");
}

void test_hush_frontend(TestContext& ctx) {
  runtime::HushConfig config;
  runtime::HushFrontend frontend(config);
  const runtime::HushFeatures features = frontend.process(
      runtime::AudioFrame::mono_16k(std::vector<float>(160, 0.25f)));
  ctx.expect_true(features.spectrum.shape() == runtime::Shape({1, 161, 2}),
                  "Hush frontend should produce one complex spectrum frame");
  ctx.expect_true(features.erb.shape() == runtime::Shape({1, 32}),
                  "Hush frontend should produce 32 ERB features");
  ctx.expect_true(features.df.shape() == runtime::Shape({2, 1, 64}),
                  "Hush frontend should produce two-channel DF features");
  ctx.expect_true(std::isfinite(features.df.data<float>()[0]),
                  "Hush frontend DF features should be finite");
  ctx.expect_true(std::isfinite(features.spectrum.data<float>()[0]),
                  "Hush frontend spectrum should contain finite values");
  frontend.reset();
  bool wrong_frame_rejected = false;
  try {
    frontend.process(runtime::AudioFrame::mono_16k(std::vector<float>(80)));
  } catch (const std::invalid_argument&) {
    wrong_frame_rejected = true;
  }
  ctx.expect_true(wrong_frame_rejected, "Hush frontend should reject wrong frame length");

  runtime::CPUBackend backend;
  runtime::MemoryPool pool;
  const runtime::HushFeatures backend_features = frontend.process(
      runtime::AudioFrame::mono_16k(std::vector<float>(160, 0.1f)), backend, pool);
  ctx.expect_true(backend_features.spectrum.shape() == runtime::Shape({1, 161, 2}),
                  "Hush frontend should use backend STFT");
}

void test_erb_mask(TestContext& ctx) {
  const runtime::Tensor spectrum = runtime::Tensor::from_vector(
      {1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  const runtime::Tensor mask = runtime::Tensor::from_vector({1, 2}, {0.5f, 0.25f});
  const runtime::Tensor projection = runtime::Tensor::from_vector(
      {2, 2}, {1.0f, 0.0f, 0.0f, 2.0f});
  const runtime::Tensor output = runtime::cpu::apply_erb_mask(spectrum, mask, projection);
  ctx.expect_close(output.data<float>()[0], 0.5f, 1e-6f, "ERB mask real bin 0");
  ctx.expect_close(output.data<float>()[1], 1.0f, 1e-6f, "ERB mask imag bin 0");
  ctx.expect_close(output.data<float>()[2], 1.5f, 1e-6f, "ERB mask real bin 1");
  ctx.expect_close(output.data<float>()[3], 2.0f, 1e-6f, "ERB mask imag bin 1");
}

void test_backend_istft(TestContext& ctx) {
  std::vector<float> samples(640);
  for (std::size_t index = 0; index < samples.size(); ++index) {
    samples[index] = 0.3f * std::sin(static_cast<float>(index) * 0.11f);
  }
  const runtime::Spectrogram reference = runtime::stft(samples, 320, 160);
  std::vector<float> packed;
  for (const runtime::Complex32& value : reference.values) {
    packed.push_back(value.real);
    packed.push_back(value.imag);
  }
  runtime::CPUBackend backend;
  runtime::MemoryPool pool;
  const runtime::Tensor spectrum = runtime::Tensor::from_vector({3, 161, 2}, packed);
  const runtime::Tensor output = backend.istft(spectrum, 320, 160, samples.size(), pool);
  ctx.expect_true(output.shape() == runtime::Shape({samples.size()}), "backend ISTFT should restore length");
  ctx.expect_close(output.data<float>()[100], samples[100], 2e-4f, "backend ISTFT sample");
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
      {"mul", test_mul},
      {"matmul", test_matmul},
      {"memory_pool_reuse", test_memory_pool_reuse},
      {"graph_execution", test_graph_execution},
      {"audio_stream", test_audio_stream},
      {"weight_file", test_weight_file},
      {"spectral_round_trip", test_spectral_round_trip},
      {"onnx_weights", test_onnx_weights},
      {"hush_bundle", test_hush_bundle},
      {"hush_model", test_hush_model},
      {"shape_kernels", test_shape_kernels},
      {"hush_stream_session", test_hush_stream_session},
      {"onnx_conv_attributes", test_onnx_conv_attributes},
      {"onnx_conv_transpose", test_onnx_conv_transpose},
      {"onnx_shape_constant", test_onnx_shape_constant},
      {"onnx_execution_plan", test_onnx_execution_plan},
      {"onnx_connection_ops", test_onnx_connection_ops},
      {"onnx_slice", test_onnx_slice},
      {"onnx_einsum", test_onnx_einsum},
      {"hush_frontend", test_hush_frontend},
      {"erb_mask", test_erb_mask},
      {"backend_istft", test_backend_istft},
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
