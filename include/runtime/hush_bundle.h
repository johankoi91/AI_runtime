#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace runtime {

struct BundleEntry {
  std::string name;
  std::vector<std::uint8_t> data;
};

class HushModelBundle {
 public:
  static HushModelBundle read(const std::string& path);

  bool has(const std::string& name) const;
  const std::vector<std::uint8_t>& entry(const std::string& name) const;
  const std::vector<BundleEntry>& entries() const { return entries_; }

 private:
  explicit HushModelBundle(std::vector<BundleEntry> entries)
      : entries_(std::move(entries)) {}

  std::vector<BundleEntry> entries_;
};

}  // namespace runtime
