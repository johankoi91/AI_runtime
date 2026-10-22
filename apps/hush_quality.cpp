#include <cmath>
#include <iostream>
#include <stdexcept>

#include "runtime/audio.h"

int main(int argc, char** argv) {
  if (argc != 3) { std::cerr << "usage: hush_quality input.wav output.wav\n"; return 2; }
  try {
    const auto input = runtime::read_wav_mono16(argv[1]);
    const auto output = runtime::read_wav_mono16(argv[2]);
    if (input.size() != output.size()) throw std::runtime_error("input/output length mismatch");
    double input_sq = 0.0, output_sq = 0.0, input_mean = 0.0, output_mean = 0.0;
    float peak = 0.0f; std::size_t finite = 0, nonzero = 0;
    for (std::size_t index = 0; index < input.size(); ++index) {
      input_mean += input[index]; output_mean += output[index];
      input_sq += static_cast<double>(input[index]) * input[index];
      output_sq += static_cast<double>(output[index]) * output[index];
      peak = std::max(peak, std::fabs(output[index]));
      finite += std::isfinite(output[index]); nonzero += std::fabs(output[index]) > 1e-8f;
    }
    input_mean /= input.size(); output_mean /= output.size();
    double covariance = 0.0, input_variance = 0.0, output_variance = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index) {
      const double left = input[index] - input_mean, right = output[index] - output_mean;
      covariance += left * right; input_variance += left * left; output_variance += right * right;
    }
    const double correlation = covariance / std::sqrt(std::max(input_variance * output_variance, 1e-30));
    std::cout << "samples=" << output.size() << " finite=" << finite
              << " rms_in=" << std::sqrt(input_sq / input.size())
              << " rms_out=" << std::sqrt(output_sq / output.size())
              << " peak_out=" << peak << " nonzero_ratio="
              << static_cast<double>(nonzero) / output.size()
              << " correlation=" << correlation << "\n";
    return finite == output.size() && peak <= 1.0f ? 0 : 1;
  } catch (const std::exception& error) { std::cerr << "quality check failed: " << error.what() << "\n"; return 1; }
}
