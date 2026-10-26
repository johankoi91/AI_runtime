#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <memory>
#include <atomic>
#include <cstdlib>
#include <new>

#include "runtime/hush_model.h"
#include "runtime/hush_session.h"
#include "runtime/metal_backend.h"

#include "runtime/influxdb_sink.h"

thread_local bool count_allocations = false;
thread_local std::size_t allocation_count = 0;

void* operator new(std::size_t bytes) {
  if (count_allocations) ++allocation_count;
  if (void* memory = std::malloc(bytes == 0 ? 1 : bytes)) return memory;
  throw std::bad_alloc();
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

int main(int argc, char** argv) {
  const bool bounded = argc > 1 && std::string(argv[1]) == "bounded";
  runtime::InfluxDbConfig config;
  if (argc > 1 && std::string(argv[1]) == "stalled") {
    auto sink = std::make_unique<runtime::InfluxDbSink>(config);
    sink->emit("hush.test", "stage_start", "stalled_upload");
    std::cout << "READY\n" << std::flush;
    std::cin.get();
    std::vector<double> durations;
    durations.reserve(100000);
    const auto start = std::chrono::steady_clock::now();
    count_allocations = true;
    for (std::size_t event = 0; event < 100000; ++event) {
      const auto before = std::chrono::steady_clock::now();
      sink->emit("hush.test", "node_complete", "shape=[1,32]", 128, .25);
      durations.push_back(std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - before).count());
    }
    count_allocations = false;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::size_t hush_frames = 0;
    if (const char* bundle = std::getenv("HUSH_MODEL_BUNDLE")) {
      const auto model = runtime::HushModel::load(bundle);
      runtime::MetalBackend backend;
      runtime::HushStreamSession session(model, backend);
      session.set_trace_sink(sink.get());
      runtime::MemoryPool pool;
      std::vector<runtime::AudioFrame> frames;
      for (std::size_t frame = 0; frame < 32; ++frame)
        frames.push_back(runtime::AudioFrame::mono_16k(std::vector<float>(160, .01f)));
      const auto output = session.process_audio_batch(frames, "feat_erb", "feat_spec", "m", backend, pool);
      if (output.size() != frames.size()) return 1;
      hush_frames = output.size();
    }
    const auto status = sink->status();
    const auto retiring = std::chrono::steady_clock::now();
    sink.reset();
    const double destroy_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - retiring).count();
    std::sort(durations.begin(), durations.end());
    std::cout << "emit_ms=" << elapsed_ms << " p99_us=" << durations[99000]
              << " destroy_ms=" << destroy_ms << " dropped=" << status.dropped_events
              << " pending=" << status.pending_bytes << " emit_allocations=" << allocation_count
              << " hush_frames=" << hush_frames << "\n";
    return elapsed_ms < 1000 && durations[99000] < 1000 && destroy_ms < 100 &&
        allocation_count == 0 && status.dropped_events > 0 &&
        status.pending_bytes <= config.max_pending_bytes ? 0 : 1;
  }
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
  return status.accepted_events > 0 && status.accepted_events + status.dropped_events == 400 &&
      status.uploaded_events == status.accepted_events && status.pending_bytes == 0 ? 0 : 1;
}
