#pragma once

#include <vector>
#include <cstddef>

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
  const Tensor& erb_projection() const { return inverse_filterbank_; }
  void reset();

 private:
  static std::vector<std::size_t> make_erb_widths(const HushConfig& config);
  HushConfig config_;
  std::vector<float> history_;
  std::vector<std::size_t> erb_widths_;
  Tensor filterbank_;
  Tensor inverse_filterbank_;
  std::vector<float> erb_mean_state_;
  std::vector<float> df_unit_state_;
  float norm_alpha_ = 0.0f;
};

}  // namespace runtime
