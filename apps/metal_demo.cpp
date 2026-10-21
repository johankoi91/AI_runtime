#include <cmath>
#include <algorithm>
#include <iostream>

#include "runtime/executor.h"
#include "runtime/graph.h"
#include "runtime/influxdb_sink.h"
#include "runtime/kernels.h"
#include "runtime/memory_pool.h"
#include "runtime/metal_backend.h"
#include "runtime/spectral.h"

int main() {
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
      .name = "hidden", .op = runtime::OpType::MatMul, .inputs = {"a", "b"}});
  graph.add_node(runtime::Node{
      .name = "out", .op = runtime::OpType::Add, .inputs = {"hidden", "bias"}});

  runtime::MemoryPool pool;
  runtime::MetalBackend backend;
  runtime::InfluxDbSink trace;
  trace.emit("runtime", "run_start", "metal demo started");
  runtime::Executor executor(backend, nullptr, &trace);
  executor.run(graph, pool);

  const float* output = graph.tensor("out").data<float>();
  const float expected[] = {32.0f, 48.0f, 79.0f, 104.0f};
  for (int i = 0; i < 4; ++i) {
    if (std::fabs(output[i] - expected[i]) > 1e-5f) {
      std::cerr << "Metal output mismatch at index " << i << ": " << output[i] << "\n";
      return 1;
    }
  }

  trace.emit("runtime", "run_complete", "output verified", graph.tensor("out").nbytes());

  std::cout << "Metal backend: " << backend.name() << "\n";
  std::cout << "Trace run id: " << trace.run_id()
            << (trace.enabled() ? " (InfluxDB enabled)\n" : " (InfluxDB disabled)\n");
  std::cout << "GPU graph result: [" << output[0] << ", " << output[1] << ", "
            << output[2] << ", " << output[3] << "]\n";
  std::cout << "Metal GPU test passed\n";

  std::vector<float> waveform(640);
  for (std::size_t index = 0; index < waveform.size(); ++index) {
    waveform[index] = 0.4f * std::sin(static_cast<float>(index) * 0.17f);
  }
  runtime::Tensor waveform_tensor = runtime::Tensor::from_vector({waveform.size()}, waveform);
  runtime::Tensor gpu_spectrogram = backend.stft(waveform_tensor, 320, 160, pool);
  runtime::Spectrogram cpu_spectrogram = runtime::stft(waveform, 320, 160);
  const float* gpu_values = gpu_spectrogram.data<float>();
  float max_error = 0.0f;
  for (std::size_t index = 0; index < cpu_spectrogram.values.size() * 2; ++index) {
    const float expected = index % 2 == 0
        ? cpu_spectrogram.values[index / 2].real
        : cpu_spectrogram.values[index / 2].imag;
    max_error = std::max(max_error, std::fabs(gpu_values[index] - expected));
  }
  if (max_error > 1e-3f) {
    std::cerr << "Metal STFT mismatch: max error=" << max_error << "\n";
    return 1;
  }
  std::cout << "Metal STFT test passed (max error=" << max_error << ")\n";

  runtime::Tensor activation_input =
      runtime::Tensor::from_vector({4}, {-2.0f, -0.25f, 0.5f, 2.0f});
  runtime::Tensor gpu_sigmoid = backend.sigmoid(activation_input, pool);
  runtime::Tensor gpu_tanh = backend.tanh(activation_input, pool);
  runtime::Tensor cpu_sigmoid = runtime::cpu::sigmoid(activation_input);
  runtime::Tensor cpu_tanh = runtime::cpu::tanh(activation_input);
  const float* gpu_sigmoid_data = gpu_sigmoid.data<float>();
  const float* gpu_tanh_data = gpu_tanh.data<float>();
  const float* cpu_sigmoid_data = cpu_sigmoid.data<float>();
  const float* cpu_tanh_data = cpu_tanh.data<float>();
  max_error = 0.0f;
  for (std::size_t index = 0; index < 4; ++index) {
    max_error = std::max(max_error, std::fabs(gpu_sigmoid_data[index] - cpu_sigmoid_data[index]));
    max_error = std::max(max_error, std::fabs(gpu_tanh_data[index] - cpu_tanh_data[index]));
  }
  if (max_error > 1e-6f) {
    std::cerr << "Metal activation mismatch: max error=" << max_error << "\n";
    return 1;
  }
  std::cout << "Metal activation test passed (max error=" << max_error << ")\n";

  runtime::Tensor gru_input = runtime::Tensor::from_vector({2}, {0.3f, -0.7f});
  runtime::Tensor gru_hidden = runtime::Tensor::from_vector({3}, {0.1f, -0.2f, 0.4f});
  runtime::Tensor gru_input_weights = runtime::Tensor::from_vector(
      {9, 2}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f,
               1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f});
  runtime::Tensor gru_hidden_weights = runtime::Tensor::from_vector(
      {9, 3}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f,
               1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
               1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f});
  runtime::Tensor gru_bias = runtime::Tensor::from_vector(
      {9}, {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f});
  runtime::Tensor gpu_gru = backend.gru_step(
      gru_input, gru_hidden, gru_input_weights, gru_hidden_weights, gru_bias, pool);
  runtime::Tensor cpu_gru = runtime::cpu::gru_step(
      gru_input, gru_hidden, gru_input_weights, gru_hidden_weights, gru_bias);
  max_error = 0.0f;
  for (std::size_t index = 0; index < 3; ++index) {
    max_error = std::max(max_error, std::fabs(gpu_gru.data<float>()[index] - cpu_gru.data<float>()[index]));
  }
  if (max_error > 1e-6f) {
    std::cerr << "Metal GRU mismatch: max error=" << max_error << "\n";
    return 1;
  }
  std::cout << "Metal GRU test passed (max error=" << max_error << ")\n";
  return 0;
}
