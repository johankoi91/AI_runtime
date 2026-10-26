#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "runtime/influxdb_sink.h"

int main(int argc, char** argv) {
  const bool bounded = argc > 1 && std::string(argv[1]) == "bounded";
  runtime::InfluxDbConfig config;
  if (bounded) config.max_pending_bytes = 1024;
  runtime::InfluxDbSink sink(config);
  std::vector<std::thread> writers;
  for (std::size_t writer = 0; writer < 4; ++writer) {
    writers.emplace_back([&sink, writer] {
      for (std::size_t event = 0; event < 100; ++event)
        sink.emit("hush.test", "node_complete",
                  "writer=" + std::to_string(writer) + " shape=[1,32] quote=\"value\"\nend", 128, .25);
    });
  }
  for (auto& writer : writers) writer.join();
  const bool uploaded = sink.flush(bounded ? std::chrono::milliseconds(100) : std::chrono::seconds(10));
  const auto status = sink.status();
  std::cout << "accepted=" << status.accepted_events << " uploaded=" << status.uploaded_events
            << " dropped=" << status.dropped_events << " pending=" << status.pending_bytes << "\n";
  if (bounded)
    return !uploaded && status.dropped_events > 0 && status.pending_bytes <= 1024 ? 0 : 1;
  return uploaded && status.accepted_events == 400 && status.uploaded_events == 400 &&
      status.dropped_events == 0 && status.pending_bytes == 0 ? 0 : 1;
}
