#include "runtime/onnx_weights.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace runtime {

namespace {

struct Reader {
  const std::vector<std::uint8_t>& bytes;
  std::size_t position = 0;

  bool done() const { return position == bytes.size(); }

  std::uint64_t varint() {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 7) {
      if (done()) {
        throw std::runtime_error("truncated protobuf varint");
      }
      const std::uint8_t byte = bytes[position++];
      if (shift == 63 && byte > 1) {
        throw std::runtime_error("protobuf varint overflows uint64");
      }
      value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) {
        return value;
      }
    }
    throw std::runtime_error("protobuf varint is too long");
  }

  std::uint32_t fixed32() {
    if (bytes.size() - position < 4) {
      throw std::runtime_error("truncated protobuf fixed32");
    }
    const std::uint32_t value =
        static_cast<std::uint32_t>(bytes[position]) |
        (static_cast<std::uint32_t>(bytes[position + 1]) << 8) |
        (static_cast<std::uint32_t>(bytes[position + 2]) << 16) |
        (static_cast<std::uint32_t>(bytes[position + 3]) << 24);
    position += 4;
    return value;
  }

  std::vector<std::uint8_t> length_delimited() {
    const std::uint64_t length = varint();
    if (length > bytes.size() - position) {
      throw std::runtime_error("truncated protobuf field");
    }
    const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(position);
    std::vector<std::uint8_t> value(begin, begin + static_cast<std::ptrdiff_t>(length));
    position += static_cast<std::size_t>(length);
    return value;
  }

  void skip(unsigned wire_type) {
    switch (wire_type) {
      case 0:
        (void)varint();
        return;
      case 1:
        if (bytes.size() - position < 8) throw std::runtime_error("truncated protobuf fixed64");
        position += 8;
        return;
      case 2:
        (void)length_delimited();
        return;
      case 5:
        if (bytes.size() - position < 4) throw std::runtime_error("truncated protobuf fixed32");
        position += 4;
        return;
      default:
        throw std::runtime_error("unsupported protobuf wire type");
    }
  }
};

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("could not open ONNX model: " + path);
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size < 0 || static_cast<std::uint64_t>(size) > (1ull << 32)) {
    throw std::runtime_error("ONNX model is too large");
  }
  file.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!file) throw std::runtime_error("could not read ONNX model: " + path);
  return bytes;
}

std::vector<std::int64_t> packed_int64(const std::vector<std::uint8_t>& bytes) {
  Reader reader{bytes};
  std::vector<std::int64_t> values;
  while (!reader.done()) values.push_back(static_cast<std::int64_t>(reader.varint()));
  return values;
}

std::vector<float> packed_float(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() % sizeof(float) != 0) throw std::runtime_error("invalid packed float_data");
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::size_t numel(const std::vector<std::size_t>& shape) {
  std::size_t result = 1;
  for (std::size_t dimension : shape) {
    if (dimension != 0 && result > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::runtime_error("ONNX tensor shape overflows size_t");
    }
    result *= dimension;
  }
  return result;
}

WeightTensor parse_tensor(const std::vector<std::uint8_t>& bytes) {
  Reader reader{bytes};
  WeightTensor tensor;
  std::int32_t data_type = 0;
  std::vector<float> float_data;
  std::vector<std::uint8_t> raw_data;
  while (!reader.done()) {
    const std::uint64_t key = reader.varint();
    const unsigned field = static_cast<unsigned>(key >> 3);
    const unsigned wire = static_cast<unsigned>(key & 7);
    if (field == 1 && wire == 0) {
      tensor.shape.push_back(static_cast<std::size_t>(reader.varint()));
    } else if (field == 1 && wire == 2) {
      for (std::int64_t dimension : packed_int64(reader.length_delimited())) {
        if (dimension < 0) throw std::runtime_error("negative ONNX tensor dimension");
        tensor.shape.push_back(static_cast<std::size_t>(dimension));
      }
    } else if (field == 2 && wire == 0) {
      data_type = static_cast<std::int32_t>(reader.varint());
    } else if (field == 4 && wire == 5) {
      const std::uint32_t bits = reader.fixed32();
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      float_data.push_back(value);
    } else if (field == 4 && wire == 2) {
      const auto packed = reader.length_delimited();
      const auto values = packed_float(packed);
      float_data.insert(float_data.end(), values.begin(), values.end());
    } else if (field == 8 && wire == 2) {
      const auto name = reader.length_delimited();
      tensor.name.assign(reinterpret_cast<const char*>(name.data()), name.size());
    } else if (field == 9 && wire == 2) {
      raw_data = reader.length_delimited();
    } else {
      reader.skip(wire);
    }
  }
  if (data_type != 1) throw std::runtime_error("ONNX initializer is not float32");
  if (!raw_data.empty()) {
    float_data = packed_float(raw_data);
  }
  if (float_data.size() != numel(tensor.shape)) {
    throw std::runtime_error("ONNX initializer shape does not match float data");
  }
  tensor.values = std::move(float_data);
  return tensor;
}

std::vector<WeightTensor> parse_graph(const std::vector<std::uint8_t>& bytes) {
  Reader reader{bytes};
  std::vector<WeightTensor> tensors;
  while (!reader.done()) {
    const std::uint64_t key = reader.varint();
    const unsigned field = static_cast<unsigned>(key >> 3);
    const unsigned wire = static_cast<unsigned>(key & 7);
    if (field == 5 && wire == 2) tensors.push_back(parse_tensor(reader.length_delimited()));
    else reader.skip(wire);
  }
  return tensors;
}

}  // namespace

std::vector<WeightTensor> OnnxWeights::read_model(const std::string& path) {
  const auto model = read_file(path);
  Reader reader{model};
  std::vector<WeightTensor> tensors;
  while (!reader.done()) {
    const std::uint64_t key = reader.varint();
    const unsigned field = static_cast<unsigned>(key >> 3);
    const unsigned wire = static_cast<unsigned>(key & 7);
    if (field == 7 && wire == 2) {
      const auto graph = parse_graph(reader.length_delimited());
      tensors.insert(tensors.end(), graph.begin(), graph.end());
    } else {
      reader.skip(wire);
    }
  }
  return tensors;
}

}  // namespace runtime
