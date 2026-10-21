#pragma once

#include <string>
#include <vector>

#include "runtime/weights.h"

namespace runtime {

class OnnxWeights {
 public:
  static std::vector<WeightTensor> read_model(const std::string& path);
};

}  // namespace runtime
