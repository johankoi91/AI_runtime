#pragma once

#include <vector>

#include "runtime/hush_model.h"
#include "runtime/tensor.h"
#include "runtime/audio.h"

namespace runtime {

struct HushFeatures {
  Tensor spectrum;
  Tensor erb;
};

class HushFrontend {
 public:
  explicit HushFrontend(const HushConfig& config);

  HushFeatures process(const AudioFrame& frame);
  void reset();

 private:
  HushConfig config_;
  std::vector<float> history_;
  Tensor filterbank_;
};

}  // namespace runtime
