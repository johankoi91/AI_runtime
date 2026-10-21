#include <cmath>
#include <iostream>

#include "runtime/executor.h"
#include "runtime/graph.h"
#include "runtime/influxdb_sink.h"
#include "runtime/memory_pool.h"
#include "runtime/metal_backend.h"

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
  return 0;
}
