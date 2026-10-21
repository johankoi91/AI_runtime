#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace runtime {

struct WeightTensor {
  std::string name;
  std::vector<std::size_t> shape;
  std::vector<float> values;
};

class WeightFile {
 public:
  static void write(const std::string& path, const std::vector<WeightTensor>& tensors);
  static std::vector<WeightTensor> read(const std::string& path);
};

}  // namespace runtime
