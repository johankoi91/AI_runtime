#include "runtime/audio.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace runtime {

namespace {
std::uint16_t read_u16(std::ifstream& file) { std::uint8_t bytes[2]; file.read(reinterpret_cast<char*>(bytes), 2); return bytes[0] | (bytes[1] << 8); }
std::uint32_t read_u32(std::ifstream& file) { std::uint8_t bytes[4]; file.read(reinterpret_cast<char*>(bytes), 4); return bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24); }
void write_u16(std::ofstream& file, std::uint16_t value) { char bytes[2] = {static_cast<char>(value), static_cast<char>(value >> 8)}; file.write(bytes, 2); }
void write_u32(std::ofstream& file, std::uint32_t value) { char bytes[4] = {static_cast<char>(value), static_cast<char>(value >> 8), static_cast<char>(value >> 16), static_cast<char>(value >> 24)}; file.write(bytes, 4); }
}

std::vector<float> read_wav_mono16(const std::string& path, std::size_t expected_sample_rate) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("could not open WAV: " + path);
  char riff[4], wave[4]; file.read(riff, 4); (void)read_u32(file); file.read(wave, 4);
  if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") throw std::runtime_error("unsupported WAV header");
  std::uint16_t channels = 0, bits = 0; std::uint32_t sample_rate = 0, data_size = 0; std::streampos data_pos{};
  while (file && !file.eof()) {
    char id[4]; file.read(id, 4); if (file.gcount() != 4) break; const std::uint32_t size = read_u32(file);
    if (std::string(id, 4) == "fmt ") { const auto format = read_u16(file); channels = read_u16(file); sample_rate = read_u32(file); (void)read_u32(file); (void)read_u16(file); bits = read_u16(file); if (size > 16) file.seekg(size - 16, std::ios::cur); if (format != 1) throw std::runtime_error("WAV must be PCM"); }
    else if (std::string(id, 4) == "data") { data_pos = file.tellg(); data_size = size; file.seekg(size, std::ios::cur); }
    else file.seekg(size, std::ios::cur);
  }
  if (channels != 1 || bits != 16 || sample_rate != expected_sample_rate || data_pos == std::streampos{}) throw std::runtime_error("WAV must be mono PCM16 at expected sample rate");
  file.clear(); file.seekg(data_pos); std::vector<float> samples(data_size / 2);
  for (float& sample : samples) sample = static_cast<float>(static_cast<std::int16_t>(read_u16(file))) / 32768.0f;
  return samples;
}

void write_wav_mono16(const std::string& path, const std::vector<float>& samples, std::size_t sample_rate) {
  if (sample_rate == 0 || samples.size() > (std::numeric_limits<std::uint32_t>::max() - 44) / 2) throw std::invalid_argument("invalid WAV output");
  std::ofstream file(path, std::ios::binary | std::ios::trunc); if (!file) throw std::runtime_error("could not write WAV: " + path);
  const std::uint32_t data_size = static_cast<std::uint32_t>(samples.size() * 2);
  file.write("RIFF", 4); write_u32(file, 36 + data_size); file.write("WAVEfmt ", 8); write_u32(file, 16); write_u16(file, 1); write_u16(file, 1); write_u32(file, sample_rate); write_u32(file, sample_rate * 2); write_u16(file, 2); write_u16(file, 16); file.write("data", 4); write_u32(file, data_size);
  for (float sample : samples) { const float clipped = std::max(-1.0f, std::min(0.999969f, sample)); write_u16(file, static_cast<std::uint16_t>(static_cast<std::int16_t>(clipped * 32768.0f))); }
}

}  // namespace runtime
