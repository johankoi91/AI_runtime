#include "runtime/weights.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace runtime {

namespace {

constexpr char kMagic[] = "MRTW0001";
constexpr std::uint32_t kMaxTensors = 1'000'000;
constexpr std::uint32_t kMaxRank = 32;
constexpr std::uint32_t kMaxNameLength = 1 << 20;
constexpr std::uint64_t kMaxValues = 1ull << 32;

template <typename T>
void write_value(std::ofstream& file, T value) {
  file.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!file) {
    throw std::runtime_error("weight file write failed");
  }
}

template <typename T>
T read_value(std::ifstream& file) {
  T value{};
  file.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!file) {
    throw std::runtime_error("weight file is truncated");
  }
  return value;
}

std::size_t checked_numel(const std::vector<std::size_t>& shape) {
  std::size_t numel = 1;
  for (std::size_t dimension : shape) {
    if (dimension != 0 && numel > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::runtime_error("weight tensor shape overflows size_t");
    }
    numel *= dimension;
  }
  return numel;
}

}  // namespace

void WeightFile::write(const std::string& path, const std::vector<WeightTensor>& tensors) {
  if (tensors.size() > kMaxTensors) {
    throw std::invalid_argument("too many tensors in weight file");
  }

  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    throw std::runtime_error("could not open weight file for writing: " + path);
  }
  file.write(kMagic, sizeof(kMagic) - 1);
  write_value<std::uint32_t>(file, static_cast<std::uint32_t>(tensors.size()));

  for (const WeightTensor& tensor : tensors) {
    if (tensor.name.size() > kMaxNameLength || tensor.shape.size() > kMaxRank) {
      throw std::invalid_argument("weight tensor metadata is too large");
    }
    if (checked_numel(tensor.shape) != tensor.values.size() ||
        tensor.values.size() > kMaxValues) {
      throw std::invalid_argument("weight tensor shape does not match values");
    }
    write_value<std::uint32_t>(file, static_cast<std::uint32_t>(tensor.name.size()));
    write_value<std::uint32_t>(file, static_cast<std::uint32_t>(tensor.shape.size()));
    write_value<std::uint64_t>(file, static_cast<std::uint64_t>(tensor.values.size()));
    file.write(tensor.name.data(), static_cast<std::streamsize>(tensor.name.size()));
    for (std::size_t dimension : tensor.shape) {
      write_value<std::uint64_t>(file, static_cast<std::uint64_t>(dimension));
    }
    file.write(
        reinterpret_cast<const char*>(tensor.values.data()),
        static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    if (!file) {
      throw std::runtime_error("weight file write failed");
    }
  }
}

std::vector<WeightTensor> WeightFile::read(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("could not open weight file: " + path);
  }

  char magic[sizeof(kMagic) - 1]{};
  file.read(magic, sizeof(magic));
  if (!file || std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(kMagic) - 1)) {
    throw std::runtime_error("unsupported weight file format");
  }

  const std::uint32_t tensor_count = read_value<std::uint32_t>(file);
  if (tensor_count > kMaxTensors) {
    throw std::runtime_error("weight file contains too many tensors");
  }

  std::vector<WeightTensor> tensors;
  tensors.reserve(tensor_count);
  for (std::uint32_t index = 0; index < tensor_count; ++index) {
    const std::uint32_t name_length = read_value<std::uint32_t>(file);
    const std::uint32_t rank = read_value<std::uint32_t>(file);
    const std::uint64_t value_count = read_value<std::uint64_t>(file);
    if (name_length > kMaxNameLength || rank > kMaxRank || value_count > kMaxValues) {
      throw std::runtime_error("weight tensor metadata exceeds safety limits");
    }

    WeightTensor tensor;
    tensor.name.resize(name_length);
    file.read(tensor.name.data(), static_cast<std::streamsize>(name_length));
    if (!file) {
      throw std::runtime_error("weight file is truncated in tensor name");
    }
    tensor.shape.resize(rank);
    for (std::size_t& dimension : tensor.shape) {
      dimension = static_cast<std::size_t>(read_value<std::uint64_t>(file));
    }
    if (checked_numel(tensor.shape) != value_count) {
      throw std::runtime_error("weight tensor shape does not match value count");
    }
    tensor.values.resize(static_cast<std::size_t>(value_count));
    file.read(
        reinterpret_cast<char*>(tensor.values.data()),
        static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    if (!file) {
      throw std::runtime_error("weight file is truncated in tensor values");
    }
    tensors.push_back(std::move(tensor));
  }
  return tensors;
}

}  // namespace runtime
