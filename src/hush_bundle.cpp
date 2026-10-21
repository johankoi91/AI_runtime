#include "runtime/hush_bundle.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <zlib.h>

namespace runtime {

namespace {

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("could not open Hush bundle: " + path);
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size < 0 || static_cast<std::uint64_t>(size) > (1ull << 32)) {
    throw std::runtime_error("Hush bundle is too large");
  }
  file.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  file.read(reinterpret_cast<char*>(data.data()), size);
  if (!file) throw std::runtime_error("could not read Hush bundle");
  return data;
}

std::vector<std::uint8_t> gunzip(const std::vector<std::uint8_t>& compressed) {
  z_stream stream{};
  stream.next_in = const_cast<Bytef*>(compressed.data());
  stream.avail_in = static_cast<uInt>(compressed.size());
  if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) {
    throw std::runtime_error("could not initialize gzip decoder");
  }
  std::vector<std::uint8_t> output;
  std::uint8_t buffer[64 * 1024];
  int result = Z_OK;
  while (result == Z_OK) {
    stream.next_out = buffer;
    stream.avail_out = sizeof(buffer);
    result = inflate(&stream, Z_NO_FLUSH);
    const std::size_t produced = sizeof(buffer) - stream.avail_out;
    output.insert(output.end(), buffer, buffer + produced);
  }
  inflateEnd(&stream);
  if (result != Z_STREAM_END) throw std::runtime_error("invalid gzip model bundle");
  return output;
}

std::size_t parse_octal(const std::uint8_t* field, std::size_t length) {
  std::size_t value = 0;
  for (std::size_t index = 0; index < length && field[index] != 0; ++index) {
    if (field[index] == ' ' || field[index] == '\0') continue;
    if (field[index] < '0' || field[index] > '7') throw std::runtime_error("invalid tar size");
    value = value * 8 + static_cast<std::size_t>(field[index] - '0');
  }
  return value;
}

bool zero_block(const std::uint8_t* block) {
  return std::all_of(block, block + 512, [](std::uint8_t value) { return value == 0; });
}

std::vector<BundleEntry> parse_tar(const std::vector<std::uint8_t>& tar) {
  std::vector<BundleEntry> entries;
  std::size_t offset = 0;
  while (offset + 512 <= tar.size()) {
    const std::uint8_t* header = tar.data() + offset;
    if (zero_block(header)) break;
    std::string name(reinterpret_cast<const char*>(header), 100);
    const std::size_t name_end = name.find('\0');
    name.resize(name_end == std::string::npos ? name.size() : name_end);
    const std::size_t size = parse_octal(header + 124, 12);
    const char type = static_cast<char>(header[156]);
    offset += 512;
    if (size > tar.size() - offset) throw std::runtime_error("truncated tar entry");
    if (type == '0' || type == '\0') {
      entries.push_back({name, std::vector<std::uint8_t>(
          tar.begin() + static_cast<std::ptrdiff_t>(offset),
          tar.begin() + static_cast<std::ptrdiff_t>(offset + size))});
    }
    const std::size_t padded = (size + 511) / 512 * 512;
    if (padded > tar.size() - offset) throw std::runtime_error("truncated tar padding");
    offset += padded;
  }
  return entries;
}

}  // namespace

HushModelBundle HushModelBundle::read(const std::string& path) {
  return HushModelBundle(parse_tar(gunzip(read_file(path))));
}

bool HushModelBundle::has(const std::string& name) const {
  return std::any_of(entries_.begin(), entries_.end(), [&](const BundleEntry& item) {
    return item.name == name;
  });
}

const std::vector<std::uint8_t>& HushModelBundle::entry(const std::string& name) const {
  for (const BundleEntry& item : entries_) {
    if (item.name == name) return item.data;
  }
  throw std::out_of_range("Hush bundle entry not found: " + name);
}

}  // namespace runtime
