#include "runtime/kernels.h"

#include <stdexcept>
#include <sstream>

namespace runtime::cpu {

Tensor conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t padding,
    std::size_t groups,
    MemoryPool* pool) {
  if (stride_h == 0 || stride_w == 0 || input.dtype() != DType::Float32 || weights.dtype() != DType::Float32 ||
      bias.dtype() != DType::Float32 || !input.is_contiguous() || !weights.is_contiguous() ||
      !bias.is_contiguous() || input.rank() != 4 || weights.rank() != 4 || bias.rank() != 1) {
    std::ostringstream message;
    message << "conv2d requires contiguous float32 NCHW tensors input_rank=" << input.rank()
            << " weight_rank=" << weights.rank() << " bias_rank=" << bias.rank()
            << " input_contiguous=" << input.is_contiguous()
            << " weight_contiguous=" << weights.is_contiguous()
            << " bias_contiguous=" << bias.is_contiguous();
    throw std::invalid_argument(message.str());
  }
  const std::size_t batch = input.shape()[0];
  const std::size_t input_channels = input.shape()[1];
  const std::size_t input_height = input.shape()[2];
  const std::size_t input_width = input.shape()[3];
  const std::size_t output_channels = weights.shape()[0];
  const std::size_t kernel_channels = weights.shape()[1];
  const std::size_t kernel_height = weights.shape()[2];
  const std::size_t kernel_width = weights.shape()[3];
  if (stride_h == 0 || stride_w == 0 || groups == 0 || input_channels % groups != 0 || output_channels % groups != 0 ||
      kernel_channels != input_channels / groups || bias.shape()[0] != output_channels ||
      input_height + 2 * padding < kernel_height || input_width + 2 * padding < kernel_width) {
    throw std::invalid_argument("conv2d weight shapes do not match input");
  }
  const std::size_t output_height = (input_height + 2 * padding - kernel_height) / stride_h + 1;
  const std::size_t output_width = (input_width + 2 * padding - kernel_width) / stride_w + 1;
  Tensor output = pool == nullptr
      ? Tensor::empty({batch, output_channels, output_height, output_width})
      : pool->allocate_tensor({batch, output_channels, output_height, output_width});
  const float* input_data = input.data<float>();
  const float* weight_data = weights.data<float>();
  const float* bias_data = bias.data<float>();
  float* output_data = output.data<float>();
  for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::size_t output_channel = 0; output_channel < output_channels; ++output_channel) {
      for (std::size_t output_y = 0; output_y < output_height; ++output_y) {
        for (std::size_t output_x = 0; output_x < output_width; ++output_x) {
          float value = bias_data[output_channel];
            const std::size_t group = output_channel / (output_channels / groups);
            const std::size_t channel_start = group * kernel_channels;
            for (std::size_t local_channel = 0; local_channel < kernel_channels; ++local_channel) {
              const std::size_t input_channel = channel_start + local_channel;
            for (std::size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
              for (std::size_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                const std::size_t input_y = output_y * stride_h + kernel_y;
                const std::size_t input_x = output_x * stride_w + kernel_x;
                if (input_y < padding || input_x < padding ||
                    input_y - padding >= input_height || input_x - padding >= input_width) {
                  continue;
                }
                const std::size_t input_offset =
                    ((batch_index * input_channels + input_channel) * input_height +
                     input_y - padding) * input_width + input_x - padding;
                const std::size_t weight_offset =
                    ((output_channel * kernel_channels + local_channel) * kernel_height + kernel_y) *
                    kernel_width + kernel_x;
                value += input_data[input_offset] * weight_data[weight_offset];
              }
            }
          }
          const std::size_t output_offset =
              ((batch_index * output_channels + output_channel) * output_height + output_y) *
              output_width + output_x;
          output_data[output_offset] = value;
        }
      }
    }
  }
  return output;
}

}  // namespace runtime::cpu
