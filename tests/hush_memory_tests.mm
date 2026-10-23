#import <Foundation/Foundation.h>
#include <mach/mach.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "runtime/hush_model.h"
#include "runtime/hush_session.h"
#include "runtime/metal_backend.h"

namespace {

std::uint64_t physical_bytes() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    throw std::runtime_error("could not read process memory");
  }
  return info.phys_footprint;
}

void verify_output(const runtime::AudioFrame& output) {
  if (output.sample_count() != 160) throw std::runtime_error("output length changed");
  for (float sample : output.samples()) {
    if (!std::isfinite(sample)) throw std::runtime_error("non-finite audio output");
  }
}

void exercise(const runtime::HushModel& model, runtime::MetalBackend& backend,
              std::size_t iterations, std::size_t batch_size) {
  runtime::HushStreamSession session(model, backend);
  runtime::MemoryPool pool;
  std::uint64_t warm_memory = 0;
  std::size_t warm_chunks = 0;
  std::uint64_t peak_memory = 0;
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    @autoreleasepool {
      std::vector<runtime::AudioFrame> frames;
      for (std::size_t frame_index = 0; frame_index < batch_size; ++frame_index) {
        std::vector<float> samples(160);
        for (std::size_t index = 0; index < samples.size(); ++index) {
          const std::size_t position = (iteration * batch_size + frame_index) * 160 + index;
          samples[index] = 0.1f * std::sin(static_cast<float>(position) * 0.071f) +
                           0.03f * std::sin(static_cast<float>(position) * 0.119f);
        }
        frames.push_back(runtime::AudioFrame::mono_16k(std::move(samples)));
      }
      if (batch_size == 1) {
        verify_output(session.process_audio(frames.front(), "feat_erb", "feat_spec", "m", backend, pool));
      } else {
        const auto outputs = session.process_audio_batch(frames, "feat_erb", "feat_spec", "m", backend, pool);
        if (outputs.size() != frames.size()) throw std::runtime_error("batch length changed");
        for (const auto& output : outputs) verify_output(output);
      }
      pool.reset();
      if ((iteration + 1) % 128 == 0) session.reset();
    }
    const std::uint64_t memory = physical_bytes();
    peak_memory = std::max(peak_memory, memory);
    if (iteration == 31) {
      warm_memory = memory;
      warm_chunks = pool.chunk_count();
    }
    if (warm_memory != 0 &&
        (memory > warm_memory + 64 * 1024 * 1024 || pool.chunk_count() > warm_chunks + 1)) {
      throw std::runtime_error("memory grew after warmup: bytes=" + std::to_string(memory) +
                               " chunks=" + std::to_string(pool.chunk_count()));
    }
    if ((iteration + 1) % 128 == 0 || iteration + 1 == iterations) {
      std::cout << "batch=" << batch_size << " iteration=" << iteration + 1
                << " footprint_mib=" << memory / 1048576.0
                << " chunks=" << pool.chunk_count() << std::endl;
    }
  }
  std::cout << "batch=" << batch_size << " peak_mib=" << peak_memory / 1048576.0
            << " pool_peak_bytes=" << pool.peak_bytes() << " stable=true" << std::endl;
}

}

int main() {
  const char* model_path = std::getenv("HUSH_MODEL_BUNDLE");
  if (model_path == nullptr || *model_path == '\0') {
    std::cout << "HUSH_MODEL_BUNDLE is not set; real-model memory test skipped\n";
    return 77;
  }
  @autoreleasepool {
    try {
      const runtime::HushModel model = runtime::HushModel::load(model_path);
      runtime::MetalBackend backend;
      const char* iterations_text = std::getenv("HUSH_MEMORY_ITERATIONS");
      const std::size_t iterations = iterations_text == nullptr ? 512 : std::stoul(iterations_text);
      if (iterations < 32) throw std::invalid_argument("at least 32 iterations are required");
      exercise(model, backend, iterations, 1);
      exercise(model, backend, 128, 32);
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "memory test failed: " << error.what() << std::endl;
      return 1;
    }
  }
}
