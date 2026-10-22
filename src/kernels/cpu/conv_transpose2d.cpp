#include "runtime/kernels.h"

#include <stdexcept>

namespace runtime::cpu {

Tensor conv_transpose2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride_h,
    std::size_t stride_w,
    std::size_t padding_h,
    std::size_t padding_w,
    std::size_t groups,
    MemoryPool* pool) {
  if (stride_h == 0 || stride_w == 0 || groups == 0 || input.dtype() != DType::Float32 || weights.dtype() != DType::Float32 ||
      bias.dtype() != DType::Float32 || !input.is_contiguous() || !weights.is_contiguous() ||
      !bias.is_contiguous() || input.rank() != 4 || weights.rank() != 4 || bias.rank() != 1) {
    throw std::invalid_argument("conv_transpose2d requires contiguous float32 NCHW tensors");
  }
  const std::size_t batch = input.shape()[0];
  const std::size_t input_channels = input.shape()[1];
  const std::size_t input_height = input.shape()[2];
  const std::size_t input_width = input.shape()[3];
  const std::size_t weight_input_channels = weights.shape()[0];
  const std::size_t output_channels = weights.shape()[1] * groups;
  const std::size_t kernel_height = weights.shape()[2];
  const std::size_t kernel_width = weights.shape()[3];
  if (weight_input_channels != input_channels || input_channels % groups != 0 || bias.shape()[0] != output_channels ||
      (input_height - 1) * stride_h + kernel_height < 2 * padding_h ||
      (input_width - 1) * stride_w + kernel_width < 2 * padding_w) {
    throw std::invalid_argument("conv_transpose2d weight shapes do not match input ic=" +
        std::to_string(input_channels) + " wic=" + std::to_string(weight_input_channels) +
        " wc=" + std::to_string(output_channels) + " groups=" + std::to_string(groups) +
        " bias=" + std::to_string(bias.shape()[0]) + " shape=" + std::to_string(input_height) +
        "x" + std::to_string(input_width) + " kernel=" + std::to_string(kernel_height) +
        "x" + std::to_string(kernel_width));
  }
  const std::size_t output_height = (input_height - 1) * stride_h + kernel_height - 2 * padding_h;
  const std::size_t output_width = (input_width - 1) * stride_w + kernel_width - 2 * padding_w;
  Tensor output = pool == nullptr
      ? Tensor::zeros({batch, output_channels, output_height, output_width})
      : pool->allocate_tensor({batch, output_channels, output_height, output_width});
  if (pool != nullptr) {
    output.fill(0.0f);
  }
  float* output_data = output.data<float>();
  for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
      for (std::size_t input_channel = 0; input_channel < input_channels; ++input_channel) {
      for (std::size_t input_y = 0; input_y < input_height; ++input_y) {
        for (std::size_t input_x = 0; input_x < input_width; ++input_x) {
          const std::size_t input_offset =
              ((batch_index * input_channels + input_channel) * input_height + input_y) *
              input_width + input_x;
          const float input_value = input.data<float>()[input_offset];
          for (std::size_t output_channel = 0; output_channel < output_channels; ++output_channel) {
            for (std::size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
              for (std::size_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                const std::size_t raw_y = input_y * stride_h + kernel_y;
                const std::size_t raw_x = input_x * stride_w + kernel_x;
                if (raw_y < padding_h || raw_x < padding_w ||
                    raw_y - padding_h >= output_height || raw_x - padding_w >= output_width) {
                  continue;
                }
                const std::size_t output_offset =
                    ((batch_index * output_channels + output_channel) * output_height +
                    raw_y - padding_h) * output_width + raw_x - padding_w;
                const std::size_t weight_offset =
                    ((input_channel * (output_channels / groups) + (output_channel % (output_channels / groups))) * kernel_height + kernel_y) *
                    kernel_width + kernel_x;
                output_data[output_offset] += input_value * weights.data<float>()[weight_offset];
              }
            }
          }
        }
      }
    }
  }
  for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (std::size_t output_channel = 0; output_channel < output_channels; ++output_channel) {
      for (std::size_t output_y = 0; output_y < output_height; ++output_y) {
        for (std::size_t output_x = 0; output_x < output_width; ++output_x) {
          const std::size_t offset =
              ((batch_index * output_channels + output_channel) * output_height + output_y) *
              output_width + output_x;
          output_data[offset] += bias.data<float>()[output_channel];
        }
      }
    }
  }
  return output;
}

}  // namespace runtime::cpu
