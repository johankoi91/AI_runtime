#pragma once

#include <vector>

#include "runtime/hush_model.h"
#include "runtime/tensor.h"
#include "runtime/audio.h"
#include "runtime/backend.h"

namespace runtime {

struct HushFeatures {
  Tensor spectrum;
  Tensor erb;
  Tensor df;
};

class HushFrontend {
 public:
  explicit HushFrontend(const HushConfig& config);

  HushFeatures process(const AudioFrame& frame);
  HushFeatures process(
      const AudioFrame& frame,
      const Backend& backend,
      MemoryPool& pool);
  void reset();

 private:
  HushConfig config_;
  std::vector<float> history_;
  Tensor filterbank_;
};

}  // namespace runtime
