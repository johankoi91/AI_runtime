#include "runtime/metal_backend.h"
#include "runtime/kernels.h"

#import <Metal/Metal.h>

#include <cstring>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace runtime {

namespace {

constexpr char kMetalSource[] = R"metal(
#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
  uint m;
  uint k;
  uint n;
};

kernel void add_kernel(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    out[index] = lhs[index] + rhs[index];
  }
}

kernel void mul_kernel(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) out[index] = lhs[index] * rhs[index];
}

kernel void matmul_kernel(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]) {
  const uint row = index.y;
  const uint col = index.x;
  if (row >= params.m || col >= params.n) {
    return;
  }

  float sum = 0.0f;
  for (uint p = 0; p < params.k; ++p) {
    sum += lhs[row * params.k + p] * rhs[p * params.n + col];
  }
  out[row * params.n + col] = sum;
}

kernel void stft_kernel(
    device const float* samples [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& sample_count [[buffer(2)]],
    constant uint& fft_size [[buffer(3)]],
    constant uint& hop_size [[buffer(4)]],
    constant uint& bins [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const uint frame = index / bins;
  const uint bin = index % bins;
  const float pi = 3.14159265358979323846f;
  float real_sum = 0.0f;
  float imag_sum = 0.0f;
  for (uint sample_index = 0; sample_index < fft_size; ++sample_index) {
    const uint input_index = frame * hop_size + sample_index;
    const float sample = input_index < sample_count ? samples[input_index] : 0.0f;
    const float window = 0.5f * (1.0f - cos(2.0f * pi * sample_index / fft_size));
    const float angle = -2.0f * pi * bin * sample_index / fft_size;
    real_sum += sample * window * cos(angle);
    imag_sum += sample * window * sin(angle);
  }
  const uint output_index = (frame * bins + bin) * 2;
  output[output_index] = real_sum;
  output[output_index + 1] = imag_sum;
}

kernel void sigmoid_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    output[index] = 1.0f / (1.0f + exp(-input[index]));
  }
}

kernel void tanh_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    output[index] = tanh(input[index]);
  }
}

kernel void gru_step_kernel(
    device const float* input [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device const float* input_weights [[buffer(2)]],
    device const float* hidden_weights [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& input_size [[buffer(6)]],
    constant uint& hidden_size [[buffer(7)]],
    uint unit [[thread_position_in_grid]]) {
  if (unit >= hidden_size) {
    return;
  }
  float update = bias[unit];
  float reset = bias[hidden_size + unit];
  float candidate = bias[2 * hidden_size + unit];
  float recurrent_candidate = 0.0f;
  for (uint index = 0; index < input_size; ++index) {
    update += input_weights[unit * input_size + index] * input[index];
    reset += input_weights[(hidden_size + unit) * input_size + index] * input[index];
    candidate += input_weights[(2 * hidden_size + unit) * input_size + index] * input[index];
  }
  for (uint index = 0; index < hidden_size; ++index) {
    update += hidden_weights[unit * hidden_size + index] * hidden[index];
    reset += hidden_weights[(hidden_size + unit) * hidden_size + index] * hidden[index];
    recurrent_candidate +=
        hidden_weights[(2 * hidden_size + unit) * hidden_size + index] * hidden[index];
  }
  update = 1.0f / (1.0f + exp(-update));
  reset = 1.0f / (1.0f + exp(-reset));
  candidate = tanh(candidate + reset * recurrent_candidate);
  output[unit] = update * hidden[unit] + (1.0f - update) * candidate;
}

kernel void conv2d_kernel(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& input_channels [[buffer(5)]],
    constant uint& input_height [[buffer(6)]],
    constant uint& input_width [[buffer(7)]],
    constant uint& output_channels [[buffer(8)]],
    constant uint& kernel_height [[buffer(9)]],
    constant uint& kernel_width [[buffer(10)]],
    constant uint& output_height [[buffer(11)]],
    constant uint& output_width [[buffer(12)]],
    constant uint& stride [[buffer(13)]],
    constant uint& padding [[buffer(14)]],
    constant uint& groups [[buffer(15)]],
    uint index [[thread_position_in_grid]]) {
  const uint total = batch * output_channels * output_height * output_width;
  if (index >= total) {
    return;
  }
  uint remainder = index;
  const uint output_x = remainder % output_width;
  remainder /= output_width;
  const uint output_y = remainder % output_height;
  remainder /= output_height;
  const uint output_channel = remainder % output_channels;
  const uint batch_index = remainder / output_channels;
  float value = bias[output_channel];
  const uint channels_per_group = input_channels / groups;
  const uint group = output_channel / (output_channels / groups);
  for (uint local_channel = 0; local_channel < channels_per_group; ++local_channel) {
    const uint input_channel = group * channels_per_group + local_channel;
    for (uint kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
      for (uint kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
        const uint input_y = output_y * stride + kernel_y;
        const uint input_x = output_x * stride + kernel_x;
        if (input_y < padding || input_x < padding ||
            input_y - padding >= input_height || input_x - padding >= input_width) {
          continue;
        }
        const uint input_offset =
            ((batch_index * input_channels + input_channel) * input_height +
             input_y - padding) * input_width + input_x - padding;
        const uint weight_offset =
            ((output_channel * channels_per_group + local_channel) * kernel_height + kernel_y) *
            kernel_width + kernel_x;
        value += input[input_offset] * weights[weight_offset];
      }
    }
  }
  output[index] = value;
}

kernel void conv_transpose2d_kernel(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& input_channels [[buffer(5)]],
    constant uint& input_height [[buffer(6)]],
    constant uint& input_width [[buffer(7)]],
    constant uint& output_channels [[buffer(8)]],
    constant uint& kernel_height [[buffer(9)]],
    constant uint& kernel_width [[buffer(10)]],
    constant uint& output_height [[buffer(11)]],
    constant uint& output_width [[buffer(12)]],
    constant uint& stride [[buffer(13)]],
    constant uint& padding [[buffer(14)]],
    uint index [[thread_position_in_grid]]) {
  const uint total = batch * output_channels * output_height * output_width;
  if (index >= total) {
    return;
  }
  uint remainder = index;
  const uint output_x = remainder % output_width;
  remainder /= output_width;
  const uint output_y = remainder % output_height;
  remainder /= output_height;
  const uint output_channel = remainder % output_channels;
  const uint batch_index = remainder / output_channels;
  float value = bias[output_channel];
  for (uint input_channel = 0; input_channel < input_channels; ++input_channel) {
    for (uint kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
      for (uint kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
        const int raw_y = int(output_y) + int(padding) - int(kernel_y);
        const int raw_x = int(output_x) + int(padding) - int(kernel_x);
        if (raw_y < 0 || raw_x < 0 || raw_y % int(stride) != 0 || raw_x % int(stride) != 0) {
          continue;
        }
        const uint input_y = uint(raw_y) / stride;
        const uint input_x = uint(raw_x) / stride;
        if (input_y >= input_height || input_x >= input_width) {
          continue;
        }
        const uint input_offset =
            ((batch_index * input_channels + input_channel) * input_height + input_y) *
            input_width + input_x;
        const uint weight_offset =
            ((input_channel * output_channels + output_channel) * kernel_height + kernel_y) *
            kernel_width + kernel_x;
        value += input[input_offset] * weights[weight_offset];
      }
    }
  }
  output[index] = value;
}

kernel void erb_filterbank_kernel(
    device const float* spectrum [[buffer(0)]],
    device const float* filterbank [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& frames [[buffer(3)]],
    constant uint& bins [[buffer(4)]],
    constant uint& bands [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= frames * bands) {
    return;
  }
  const uint frame = index / bands;
  const uint band = index % bands;
  float energy = 0.0f;
  for (uint bin = 0; bin < bins; ++bin) {
    const uint spectrum_offset = (frame * bins + bin) * 2;
    const float real = spectrum[spectrum_offset];
    const float imag = spectrum[spectrum_offset + 1];
    energy += filterbank[band * bins + bin] * (real * real + imag * imag);
  }
  output[index] = log(max(energy, 1e-12f));
}

kernel void exponential_unit_norm_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    output[index] = input[index] * scale;
  }
}

kernel void relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = max(input[index], 0.0f);
}

kernel void istft_kernel(
    device const float* spectrum [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& frames [[buffer(2)]],
    constant uint& fft_size [[buffer(3)]],
    constant uint& hop_size [[buffer(4)]],
    constant uint& bins [[buffer(5)]],
    constant uint& original_samples [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= original_samples) return;
  const float pi = 3.14159265358979323846f;
  float value = 0.0f;
  float energy = 0.0f;
  for (uint frame = 0; frame < frames; ++frame) {
    const int local = int(index) - int(frame * hop_size);
    if (local < 0 || uint(local) >= fft_size) continue;
    const float window = 0.5f * (1.0f - cos(2.0f * pi * local / fft_size));
    float inverse = 0.0f;
    for (uint bin = 0; bin < fft_size; ++bin) {
      float real = 0.0f;
      float imag = 0.0f;
      if (bin < bins) {
        real = spectrum[(frame * bins + bin) * 2];
        imag = spectrum[(frame * bins + bin) * 2 + 1];
      } else {
        const uint mirror = fft_size - bin;
        if (mirror == 0 || mirror >= bins - 1) continue;
        real = spectrum[(frame * bins + mirror) * 2];
        imag = -spectrum[(frame * bins + mirror) * 2 + 1];
      }
      const float angle = 2.0f * pi * bin * local / fft_size;
      inverse += real * cos(angle) - imag * sin(angle);
    }
    value += inverse / fft_size * window;
    energy += window * window;
  }
  output[index] = energy > 1e-8f ? value / energy : 0.0f;
}
)metal";

void check_status(id<MTLCommandBuffer> command_buffer, const char* operation) {
  [command_buffer waitUntilCompleted];
  if (command_buffer.status == MTLCommandBufferStatusError) {
    NSString* message = command_buffer.error.localizedDescription;
    throw std::runtime_error(
        std::string(operation) + ": " + (message ? message.UTF8String : "unknown Metal error"));
  }
}

id<MTLBuffer> make_input_buffer(id<MTLDevice> device, const Tensor& tensor) {
  return [device newBufferWithBytes:tensor.data<float>()
                             length:tensor.nbytes()
                            options:MTLResourceStorageModeShared];
}

void copy_output(id<MTLBuffer> buffer, Tensor& output) {
  std::memcpy(output.data<float>(), [buffer contents], output.nbytes());
}

}  // namespace

struct MetalBackend::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> add_pipeline = nil;
  id<MTLComputePipelineState> mul_pipeline = nil;
  id<MTLComputePipelineState> matmul_pipeline = nil;
  id<MTLComputePipelineState> stft_pipeline = nil;
  id<MTLComputePipelineState> sigmoid_pipeline = nil;
  id<MTLComputePipelineState> tanh_pipeline = nil;
  id<MTLComputePipelineState> relu_pipeline = nil;
  id<MTLComputePipelineState> gru_pipeline = nil;
  id<MTLComputePipelineState> conv2d_pipeline = nil;
  id<MTLComputePipelineState> conv_transpose2d_pipeline = nil;
  id<MTLComputePipelineState> erb_pipeline = nil;
  id<MTLComputePipelineState> norm_pipeline = nil;
  id<MTLComputePipelineState> istft_pipeline = nil;
  TraceSink* trace = nullptr;
};

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {
  impl_->device = MTLCreateSystemDefaultDevice();
  if (impl_->device == nil) {
    throw std::runtime_error("MetalBackend: no Metal device is available");
  }

  impl_->queue = [impl_->device newCommandQueue];
  if (impl_->queue == nil) {
    throw std::runtime_error("MetalBackend: failed to create command queue");
  }

  NSError* error = nil;
  NSString* source = [NSString stringWithUTF8String:kMetalSource];
  id<MTLLibrary> library = [impl_->device newLibraryWithSource:source options:nil error:&error];
  if (library == nil) {
    throw std::runtime_error(
        std::string("MetalBackend: failed to compile Metal shaders: ") +
        (error.localizedDescription ? error.localizedDescription.UTF8String : "unknown error"));
  }

  id<MTLFunction> add_function = [library newFunctionWithName:@"add_kernel"];
  id<MTLFunction> mul_function = [library newFunctionWithName:@"mul_kernel"];
  id<MTLFunction> matmul_function = [library newFunctionWithName:@"matmul_kernel"];
  id<MTLFunction> stft_function = [library newFunctionWithName:@"stft_kernel"];
  id<MTLFunction> sigmoid_function = [library newFunctionWithName:@"sigmoid_kernel"];
  id<MTLFunction> tanh_function = [library newFunctionWithName:@"tanh_kernel"];
  id<MTLFunction> relu_function = [library newFunctionWithName:@"relu_kernel"];
  id<MTLFunction> gru_function = [library newFunctionWithName:@"gru_step_kernel"];
  id<MTLFunction> conv2d_function = [library newFunctionWithName:@"conv2d_kernel"];
  id<MTLFunction> conv_transpose2d_function =
      [library newFunctionWithName:@"conv_transpose2d_kernel"];
  id<MTLFunction> erb_function = [library newFunctionWithName:@"erb_filterbank_kernel"];
  id<MTLFunction> norm_function = [library newFunctionWithName:@"exponential_unit_norm_kernel"];
  id<MTLFunction> istft_function = [library newFunctionWithName:@"istft_kernel"];
  impl_->add_pipeline = [impl_->device newComputePipelineStateWithFunction:add_function error:&error];
  if (impl_->add_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create add pipeline");
  }
  impl_->mul_pipeline = [impl_->device newComputePipelineStateWithFunction:mul_function error:&error];
  if (impl_->mul_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create mul pipeline");
  }
  impl_->matmul_pipeline =
      [impl_->device newComputePipelineStateWithFunction:matmul_function error:&error];
  if (impl_->matmul_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create matmul pipeline");
  }
  impl_->stft_pipeline =
      [impl_->device newComputePipelineStateWithFunction:stft_function error:&error];
  if (impl_->stft_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create stft pipeline");
  }
  impl_->sigmoid_pipeline =
      [impl_->device newComputePipelineStateWithFunction:sigmoid_function error:&error];
  impl_->tanh_pipeline =
      [impl_->device newComputePipelineStateWithFunction:tanh_function error:&error];
  if (impl_->sigmoid_pipeline == nil || impl_->tanh_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create activation pipelines");
  }
  impl_->relu_pipeline =
      [impl_->device newComputePipelineStateWithFunction:relu_function error:&error];
  if (impl_->relu_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create relu pipeline");
  }
  impl_->gru_pipeline =
      [impl_->device newComputePipelineStateWithFunction:gru_function error:&error];
  if (impl_->gru_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create gru pipeline");
  }
  impl_->conv2d_pipeline =
      [impl_->device newComputePipelineStateWithFunction:conv2d_function error:&error];
  if (impl_->conv2d_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create conv2d pipeline");
  }
  impl_->conv_transpose2d_pipeline = [impl_->device
      newComputePipelineStateWithFunction:conv_transpose2d_function error:&error];
  if (impl_->conv_transpose2d_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create conv transpose pipeline");
  }
  impl_->erb_pipeline =
      [impl_->device newComputePipelineStateWithFunction:erb_function error:&error];
  impl_->norm_pipeline =
      [impl_->device newComputePipelineStateWithFunction:norm_function error:&error];
  if (impl_->erb_pipeline == nil || impl_->norm_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create ERB pipelines");
  }
  impl_->istft_pipeline =
      [impl_->device newComputePipelineStateWithFunction:istft_function error:&error];
  if (impl_->istft_pipeline == nil) {
    throw std::runtime_error("MetalBackend: failed to create istft pipeline");
  }
}

MetalBackend::~MetalBackend() = default;

std::string_view MetalBackend::name() const {
  return "metal";
}

void MetalBackend::set_trace_sink(TraceSink* sink) const {
  impl_->trace = sink;
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "device_ready", "Apple Metal device=" +
        std::string(impl_->device.name.UTF8String));
    impl_->trace->emit("metal", "pipeline_ready", "matmul_kernel/add_kernel/stft_kernel compiled");
  }
}

Tensor MetalBackend::add(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const {
  if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32) {
    throw std::runtime_error("metal::add only supports float32");
  }
  if (lhs.shape() != rhs.shape()) {
    throw std::runtime_error("metal::add requires matching shapes");
  }
  if (!lhs.is_contiguous() || !rhs.is_contiguous()) {
    throw std::runtime_error("metal::add currently requires contiguous tensors");
  }

  Tensor output = pool.allocate_tensor(lhs.shape(), lhs.dtype());
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "buffer_allocated", "add output", output.nbytes());
  }
  id<MTLBuffer> lhs_buffer = make_input_buffer(impl_->device, lhs);
  id<MTLBuffer> rhs_buffer = make_input_buffer(impl_->device, rhs);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const uint32_t count = static_cast<uint32_t>(lhs.numel());

  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "input_buffer_created", "add lhs", lhs.nbytes());
    impl_->trace->emit("metal", "input_buffer_created", "add rhs", rhs.nbytes());
    impl_->trace->emit("metal", "output_buffer_created", "add output", output.nbytes());
  }

  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "command_buffer_created", "add command buffer");
  }
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->add_pipeline];
  [encoder setBuffer:lhs_buffer offset:0 atIndex:0];
  [encoder setBuffer:rhs_buffer offset:0 atIndex:1];
  [encoder setBuffer:output_buffer offset:0 atIndex:2];
  [encoder setBytes:&count length:sizeof(count) atIndex:3];
  const NSUInteger threads = std::min<NSUInteger>(impl_->add_pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "encoder_configured", "add buffers=lhs,rhs,out,count");
    impl_->trace->emit("metal", "dispatch_configured", "add grid=4 threads_per_group=" +
        std::to_string(threads));
    impl_->trace->emit("metal", "kernel_submitted", "add_kernel grid=4");
  }
  const auto start = std::chrono::steady_clock::now();
  [command_buffer commit];
  check_status(command_buffer, "metal::add");
  const auto elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "kernel_completed", "add_kernel", output.nbytes(), elapsed);
    impl_->trace->emit("metal", "copy_back_started", "add output", output.nbytes());
  }
  copy_output(output_buffer, output);
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "copy_back_completed", "add output to Tensor", output.nbytes());
  }
  return output;
}

Tensor MetalBackend::mul(const Tensor& lhs, const Tensor& rhs, MemoryPool& pool) const {
  if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32 ||
      lhs.shape() != rhs.shape() || !lhs.is_contiguous() || !rhs.is_contiguous()) {
    throw std::invalid_argument("metal::mul requires matching contiguous float32 tensors");
  }
  Tensor output = pool.allocate_tensor(lhs.shape(), lhs.dtype());
  id<MTLBuffer> lhs_buffer = make_input_buffer(impl_->device, lhs);
  id<MTLBuffer> rhs_buffer = make_input_buffer(impl_->device, rhs);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const uint32_t count = static_cast<uint32_t>(lhs.numel());
  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->mul_pipeline];
  [encoder setBuffer:lhs_buffer offset:0 atIndex:0];
  [encoder setBuffer:rhs_buffer offset:0 atIndex:1];
  [encoder setBuffer:output_buffer offset:0 atIndex:2];
  [encoder setBytes:&count length:sizeof(count) atIndex:3];
  const NSUInteger threads = std::min<NSUInteger>(impl_->mul_pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  check_status(command_buffer, "metal::mul");
  copy_output(output_buffer, output);
  return output;
}

Tensor MetalBackend::matmul(
    const Tensor& lhs,
    const Tensor& rhs,
    MemoryPool& pool) const {
  if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32) {
    throw std::runtime_error("metal::matmul only supports float32");
  }
  if (lhs.rank() != 2 || rhs.rank() != 2) {
    throw std::runtime_error("metal::matmul requires rank-2 tensors");
  }
  if (!lhs.is_contiguous() || !rhs.is_contiguous()) {
    throw std::runtime_error("metal::matmul currently requires contiguous tensors");
  }

  const uint32_t m = static_cast<uint32_t>(lhs.shape()[0]);
  const uint32_t k = static_cast<uint32_t>(lhs.shape()[1]);
  const uint32_t rhs_k = static_cast<uint32_t>(rhs.shape()[0]);
  const uint32_t n = static_cast<uint32_t>(rhs.shape()[1]);
  if (k != rhs_k) {
    throw std::runtime_error("metal::matmul inner dimensions must match");
  }

  Tensor output = pool.allocate_tensor({m, n}, lhs.dtype());
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "buffer_allocated", "matmul output", output.nbytes());
  }
  id<MTLBuffer> lhs_buffer = make_input_buffer(impl_->device, lhs);
  id<MTLBuffer> rhs_buffer = make_input_buffer(impl_->device, rhs);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const struct {
    uint32_t m;
    uint32_t k;
    uint32_t n;
  } params{m, k, n};

  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "input_buffer_created", "matmul lhs", lhs.nbytes());
    impl_->trace->emit("metal", "input_buffer_created", "matmul rhs", rhs.nbytes());
    impl_->trace->emit("metal", "output_buffer_created", "matmul output", output.nbytes());
  }

  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "command_buffer_created", "matmul command buffer");
  }
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->matmul_pipeline];
  [encoder setBuffer:lhs_buffer offset:0 atIndex:0];
  [encoder setBuffer:rhs_buffer offset:0 atIndex:1];
  [encoder setBuffer:output_buffer offset:0 atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  const NSUInteger width = std::min<NSUInteger>(impl_->matmul_pipeline.threadExecutionWidth, n);
  const NSUInteger height = std::min<NSUInteger>(
      impl_->matmul_pipeline.maxTotalThreadsPerThreadgroup / width, m);
  [encoder dispatchThreads:MTLSizeMake(n, m, 1)
      threadsPerThreadgroup:MTLSizeMake(width, height, 1)];
  [encoder endEncoding];
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "encoder_configured", "matmul buffers=lhs,rhs,out,params");
    impl_->trace->emit("metal", "dispatch_configured", "matmul grid=" +
        std::to_string(n) + "x" + std::to_string(m) + " threads_per_group=" +
        std::to_string(width) + "x" + std::to_string(height));
    impl_->trace->emit(
        "metal", "kernel_submitted",
        "matmul_kernel grid=" + std::to_string(m) + "x" + std::to_string(n));
  }
  const auto start = std::chrono::steady_clock::now();
  [command_buffer commit];
  check_status(command_buffer, "metal::matmul");
  const auto elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "kernel_completed", "matmul_kernel", output.nbytes(), elapsed);
    impl_->trace->emit("metal", "copy_back_started", "matmul output", output.nbytes());
  }
  copy_output(output_buffer, output);
  if (impl_->trace != nullptr) {
    impl_->trace->emit("metal", "copy_back_completed", "matmul output to Tensor", output.nbytes());
  }
  return output;
}

Tensor MetalBackend::stft(
    const Tensor& samples,
    std::size_t fft_size,
    std::size_t hop_size,
    MemoryPool& pool) const {
  if (samples.dtype() != DType::Float32 || samples.rank() != 1 || !samples.is_contiguous()) {
    throw std::runtime_error("metal::stft requires a contiguous float32 vector");
  }
  if (fft_size == 0 || hop_size == 0 || hop_size > fft_size) {
    throw std::runtime_error("metal::stft requires fft_size >= hop_size > 0");
  }
  const std::size_t frames = samples.numel() <= fft_size
      ? 1
      : (samples.numel() - fft_size + hop_size - 1) / hop_size + 1;
  const std::size_t bins = fft_size / 2 + 1;
  Tensor output = pool.allocate_tensor({frames, bins, 2}, DType::Float32);
  id<MTLBuffer> input_buffer = make_input_buffer(impl_->device, samples);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const uint32_t sample_count = static_cast<uint32_t>(samples.numel());
  const uint32_t fft = static_cast<uint32_t>(fft_size);
  const uint32_t hop = static_cast<uint32_t>(hop_size);
  const uint32_t bin_count = static_cast<uint32_t>(bins);
  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->stft_pipeline];
  [encoder setBuffer:input_buffer offset:0 atIndex:0];
  [encoder setBuffer:output_buffer offset:0 atIndex:1];
  [encoder setBytes:&sample_count length:sizeof(sample_count) atIndex:2];
  [encoder setBytes:&fft length:sizeof(fft) atIndex:3];
  [encoder setBytes:&hop length:sizeof(hop) atIndex:4];
  [encoder setBytes:&bin_count length:sizeof(bin_count) atIndex:5];
  const NSUInteger count = frames * bins;
  const NSUInteger threads = std::min<NSUInteger>(
      impl_->stft_pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  check_status(command_buffer, "metal::stft");
  copy_output(output_buffer, output);
  return output;
}

namespace {

Tensor metal_activation(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    const Tensor& input,
    MemoryPool& pool,
    const char* operation) {
  if (input.dtype() != DType::Float32 || !input.is_contiguous()) {
    throw std::runtime_error(std::string("metal::") + operation +
                             " requires a contiguous float32 tensor");
  }
  Tensor output = pool.allocate_tensor(input.shape(), input.dtype());
  id<MTLBuffer> input_buffer = make_input_buffer(device, input);
  id<MTLBuffer> output_buffer =
      [device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const uint32_t count = static_cast<uint32_t>(input.numel());
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input_buffer offset:0 atIndex:0];
  [encoder setBuffer:output_buffer offset:0 atIndex:1];
  [encoder setBytes:&count length:sizeof(count) atIndex:2];
  const NSUInteger threads = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  check_status(command_buffer, operation);
  copy_output(output_buffer, output);
  return output;
}

}  // namespace

Tensor MetalBackend::sigmoid(const Tensor& input, MemoryPool& pool) const {
  return metal_activation(
      impl_->device, impl_->queue, impl_->sigmoid_pipeline, input, pool, "sigmoid");
}

Tensor MetalBackend::tanh(const Tensor& input, MemoryPool& pool) const {
  return metal_activation(
      impl_->device, impl_->queue, impl_->tanh_pipeline, input, pool, "tanh");
}

Tensor MetalBackend::relu(const Tensor& input, MemoryPool& pool) const {
  return metal_activation(
      impl_->device, impl_->queue, impl_->relu_pipeline, input, pool, "relu");
}

Tensor MetalBackend::gru_step(
    const Tensor& input,
    const Tensor& hidden,
    const Tensor& input_weights,
    const Tensor& hidden_weights,
    const Tensor& bias,
    MemoryPool& pool) const {
  if (input.dtype() != DType::Float32 || hidden.dtype() != DType::Float32 ||
      input_weights.dtype() != DType::Float32 || hidden_weights.dtype() != DType::Float32 ||
      bias.dtype() != DType::Float32 || !input.is_contiguous() || !hidden.is_contiguous() ||
      !input_weights.is_contiguous() || !hidden_weights.is_contiguous() || !bias.is_contiguous()) {
    throw std::invalid_argument("metal::gru_step requires contiguous float32 tensors");
  }
  if (input.rank() != 1 || hidden.rank() != 1 || input_weights.rank() != 2 ||
      hidden_weights.rank() != 2 || bias.rank() != 1) {
    throw std::invalid_argument("metal::gru_step has invalid tensor ranks");
  }
  const std::size_t input_size = input.shape()[0];
  const std::size_t hidden_size = hidden.shape()[0];
  if (input_weights.shape() != Shape{3 * hidden_size, input_size} ||
      hidden_weights.shape() != Shape{3 * hidden_size, hidden_size} ||
      bias.shape() != Shape{3 * hidden_size}) {
    throw std::invalid_argument("metal::gru_step weight shapes do not match input and hidden sizes");
  }
  Tensor output = pool.allocate_tensor({hidden_size}, DType::Float32);
  id<MTLBuffer> input_buffer = make_input_buffer(impl_->device, input);
  id<MTLBuffer> hidden_buffer = make_input_buffer(impl_->device, hidden);
  id<MTLBuffer> input_weights_buffer = make_input_buffer(impl_->device, input_weights);
  id<MTLBuffer> hidden_weights_buffer = make_input_buffer(impl_->device, hidden_weights);
  id<MTLBuffer> bias_buffer = make_input_buffer(impl_->device, bias);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const uint32_t input_count = static_cast<uint32_t>(input_size);
  const uint32_t hidden_count = static_cast<uint32_t>(hidden_size);
  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->gru_pipeline];
  [encoder setBuffer:input_buffer offset:0 atIndex:0];
  [encoder setBuffer:hidden_buffer offset:0 atIndex:1];
  [encoder setBuffer:input_weights_buffer offset:0 atIndex:2];
  [encoder setBuffer:hidden_weights_buffer offset:0 atIndex:3];
  [encoder setBuffer:bias_buffer offset:0 atIndex:4];
  [encoder setBuffer:output_buffer offset:0 atIndex:5];
  [encoder setBytes:&input_count length:sizeof(input_count) atIndex:6];
  [encoder setBytes:&hidden_count length:sizeof(hidden_count) atIndex:7];
  const NSUInteger threads = std::min<NSUInteger>(
      impl_->gru_pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(hidden_size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  check_status(command_buffer, "metal::gru_step");
  copy_output(output_buffer, output);
  return output;
}

Tensor MetalBackend::conv2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride,
    std::size_t padding,
    std::size_t groups,
    MemoryPool& pool) const {
  if (stride == 0 || input.dtype() != DType::Float32 || weights.dtype() != DType::Float32 ||
      bias.dtype() != DType::Float32 || !input.is_contiguous() || !weights.is_contiguous() ||
      !bias.is_contiguous() || input.rank() != 4 || weights.rank() != 4 || bias.rank() != 1) {
    throw std::invalid_argument("metal::conv2d requires contiguous float32 NCHW tensors");
  }
  const std::size_t batch = input.shape()[0];
  const std::size_t input_channels = input.shape()[1];
  const std::size_t input_height = input.shape()[2];
  const std::size_t input_width = input.shape()[3];
  const std::size_t output_channels = weights.shape()[0];
  const std::size_t kernel_channels = weights.shape()[1];
  const std::size_t kernel_height = weights.shape()[2];
  const std::size_t kernel_width = weights.shape()[3];
  if (groups == 0 || input_channels % groups != 0 || output_channels % groups != 0 ||
      kernel_channels != input_channels / groups || bias.shape()[0] != output_channels ||
      input_height + 2 * padding < kernel_height || input_width + 2 * padding < kernel_width) {
    throw std::invalid_argument("metal::conv2d weight shapes do not match input");
  }
  const std::size_t output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
  const std::size_t output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
  Tensor output = pool.allocate_tensor(
      {batch, output_channels, output_height, output_width}, DType::Float32);
  id<MTLBuffer> input_buffer = make_input_buffer(impl_->device, input);
  id<MTLBuffer> weights_buffer = make_input_buffer(impl_->device, weights);
  id<MTLBuffer> bias_buffer = make_input_buffer(impl_->device, bias);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const std::uint32_t values[] = {
      static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(input_channels),
      static_cast<std::uint32_t>(input_height), static_cast<std::uint32_t>(input_width),
      static_cast<std::uint32_t>(output_channels), static_cast<std::uint32_t>(kernel_height),
      static_cast<std::uint32_t>(kernel_width), static_cast<std::uint32_t>(output_height),
      static_cast<std::uint32_t>(output_width), static_cast<std::uint32_t>(stride),
      static_cast<std::uint32_t>(padding), static_cast<std::uint32_t>(groups)};
  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->conv2d_pipeline];
  [encoder setBuffer:input_buffer offset:0 atIndex:0];
  [encoder setBuffer:weights_buffer offset:0 atIndex:1];
  [encoder setBuffer:bias_buffer offset:0 atIndex:2];
  [encoder setBuffer:output_buffer offset:0 atIndex:3];
  for (NSUInteger index = 0; index < 12; ++index) {
    [encoder setBytes:&values[index] length:sizeof(values[index]) atIndex:4 + index];
  }
  const NSUInteger count = output.numel();
  const NSUInteger threads = std::min<NSUInteger>(
      impl_->conv2d_pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  check_status(command_buffer, "metal::conv2d");
  copy_output(output_buffer, output);
  return output;
}

Tensor MetalBackend::conv_transpose2d(
    const Tensor& input,
    const Tensor& weights,
    const Tensor& bias,
    std::size_t stride,
    std::size_t padding,
    MemoryPool& pool) const {
  if (stride == 0 || input.dtype() != DType::Float32 || weights.dtype() != DType::Float32 ||
      bias.dtype() != DType::Float32 || !input.is_contiguous() || !weights.is_contiguous() ||
      !bias.is_contiguous() || input.rank() != 4 || weights.rank() != 4 || bias.rank() != 1) {
    throw std::invalid_argument("metal::conv_transpose2d requires contiguous float32 NCHW tensors");
  }
  const std::size_t batch = input.shape()[0];
  const std::size_t input_channels = input.shape()[1];
  const std::size_t input_height = input.shape()[2];
  const std::size_t input_width = input.shape()[3];
  const std::size_t weight_input_channels = weights.shape()[0];
  const std::size_t output_channels = weights.shape()[1];
  const std::size_t kernel_height = weights.shape()[2];
  const std::size_t kernel_width = weights.shape()[3];
  if (weight_input_channels != input_channels || bias.shape()[0] != output_channels ||
      (input_height - 1) * stride + kernel_height < 2 * padding ||
      (input_width - 1) * stride + kernel_width < 2 * padding) {
    throw std::invalid_argument("metal::conv_transpose2d weight shapes do not match input");
  }
  const std::size_t output_height = (input_height - 1) * stride + kernel_height - 2 * padding;
  const std::size_t output_width = (input_width - 1) * stride + kernel_width - 2 * padding;
  Tensor output = pool.allocate_tensor(
      {batch, output_channels, output_height, output_width}, DType::Float32);
  id<MTLBuffer> input_buffer = make_input_buffer(impl_->device, input);
  id<MTLBuffer> weights_buffer = make_input_buffer(impl_->device, weights);
  id<MTLBuffer> bias_buffer = make_input_buffer(impl_->device, bias);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const std::uint32_t values[] = {
      static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(input_channels),
      static_cast<std::uint32_t>(input_height), static_cast<std::uint32_t>(input_width),
      static_cast<std::uint32_t>(output_channels), static_cast<std::uint32_t>(kernel_height),
      static_cast<std::uint32_t>(kernel_width), static_cast<std::uint32_t>(output_height),
      static_cast<std::uint32_t>(output_width), static_cast<std::uint32_t>(stride),
      static_cast<std::uint32_t>(padding)};
  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->conv_transpose2d_pipeline];
  [encoder setBuffer:input_buffer offset:0 atIndex:0];
  [encoder setBuffer:weights_buffer offset:0 atIndex:1];
  [encoder setBuffer:bias_buffer offset:0 atIndex:2];
  [encoder setBuffer:output_buffer offset:0 atIndex:3];
  for (NSUInteger index = 0; index < 11; ++index) {
    [encoder setBytes:&values[index] length:sizeof(values[index]) atIndex:4 + index];
  }
  const NSUInteger count = output.numel();
  const NSUInteger threads = std::min<NSUInteger>(
      impl_->conv_transpose2d_pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  check_status(command_buffer, "metal::conv_transpose2d");
  copy_output(output_buffer, output);
  return output;
}

Tensor MetalBackend::erb_filterbank(
    const Tensor& spectrum,
    const Tensor& filterbank,
    MemoryPool& pool) const {
  if (spectrum.dtype() != DType::Float32 || filterbank.dtype() != DType::Float32 ||
      !spectrum.is_contiguous() || !filterbank.is_contiguous() || spectrum.rank() != 3 ||
      filterbank.rank() != 2 || spectrum.shape()[2] != 2 ||
      filterbank.shape()[1] != spectrum.shape()[1]) {
    throw std::invalid_argument("metal::erb_filterbank has invalid tensor shapes");
  }
  const std::size_t frames = spectrum.shape()[0];
  const std::size_t bins = spectrum.shape()[1];
  const std::size_t bands = filterbank.shape()[0];
  Tensor output = pool.allocate_tensor({frames, bands}, DType::Float32);
  id<MTLBuffer> spectrum_buffer = make_input_buffer(impl_->device, spectrum);
  id<MTLBuffer> filterbank_buffer = make_input_buffer(impl_->device, filterbank);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const uint32_t frame_count = static_cast<uint32_t>(frames);
  const uint32_t bin_count = static_cast<uint32_t>(bins);
  const uint32_t band_count = static_cast<uint32_t>(bands);
  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->erb_pipeline];
  [encoder setBuffer:spectrum_buffer offset:0 atIndex:0];
  [encoder setBuffer:filterbank_buffer offset:0 atIndex:1];
  [encoder setBuffer:output_buffer offset:0 atIndex:2];
  [encoder setBytes:&frame_count length:sizeof(frame_count) atIndex:3];
  [encoder setBytes:&bin_count length:sizeof(bin_count) atIndex:4];
  [encoder setBytes:&band_count length:sizeof(band_count) atIndex:5];
  const NSUInteger threads = std::min<NSUInteger>(impl_->erb_pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(frames * bands, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  check_status(command_buffer, "metal::erb_filterbank");
  copy_output(output_buffer, output);
  return output;
}

Tensor MetalBackend::exponential_unit_norm(
    const Tensor& input,
    float& running_energy,
    float alpha,
    MemoryPool& pool) const {
  if (input.dtype() != DType::Float32 || !input.is_contiguous() || input.rank() != 1 ||
      !(alpha > 0.0f && alpha <= 1.0f)) {
    throw std::invalid_argument("metal::exponential_unit_norm has invalid input");
  }
  float energy = 0.0f;
  for (std::size_t index = 0; index < input.numel(); ++index) {
    const float value = input.data<float>()[index];
    energy += value * value;
  }
  energy /= static_cast<float>(std::max<std::size_t>(input.numel(), 1));
  running_energy = (1.0f - alpha) * running_energy + alpha * energy;
  const float scale = 1.0f / std::sqrt(std::max(running_energy, 1e-12f));
  Tensor output = pool.allocate_tensor(input.shape(), DType::Float32);
  id<MTLBuffer> input_buffer = make_input_buffer(impl_->device, input);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const uint32_t count = static_cast<uint32_t>(input.numel());
  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->norm_pipeline];
  [encoder setBuffer:input_buffer offset:0 atIndex:0];
  [encoder setBuffer:output_buffer offset:0 atIndex:1];
  [encoder setBytes:&count length:sizeof(count) atIndex:2];
  [encoder setBytes:&scale length:sizeof(scale) atIndex:3];
  const NSUInteger threads = std::min<NSUInteger>(impl_->norm_pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  check_status(command_buffer, "metal::exponential_unit_norm");
  copy_output(output_buffer, output);
  return output;
}

Tensor MetalBackend::istft(
    const Tensor& spectrum,
    std::size_t fft_size,
    std::size_t hop_size,
    std::size_t original_samples,
    MemoryPool& pool) const {
  if (spectrum.dtype() != DType::Float32 || !spectrum.is_contiguous() || spectrum.rank() != 3 ||
      spectrum.shape()[2] != 2 || spectrum.shape()[1] != fft_size / 2 + 1 || hop_size == 0 ||
      hop_size > fft_size) {
    throw std::invalid_argument("metal::istft has invalid spectrum shape");
  }
  const std::size_t frames = spectrum.shape()[0];
  const std::size_t bins = spectrum.shape()[1];
  Tensor output = pool.allocate_tensor({original_samples}, DType::Float32);
  id<MTLBuffer> spectrum_buffer = make_input_buffer(impl_->device, spectrum);
  id<MTLBuffer> output_buffer =
      [impl_->device newBufferWithLength:output.nbytes() options:MTLResourceStorageModeShared];
  const std::uint32_t values[] = {
      static_cast<std::uint32_t>(frames), static_cast<std::uint32_t>(fft_size),
      static_cast<std::uint32_t>(hop_size), static_cast<std::uint32_t>(bins),
      static_cast<std::uint32_t>(original_samples)};
  id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:impl_->istft_pipeline];
  [encoder setBuffer:spectrum_buffer offset:0 atIndex:0];
  [encoder setBuffer:output_buffer offset:0 atIndex:1];
  for (NSUInteger index = 0; index < 5; ++index) {
    [encoder setBytes:&values[index] length:sizeof(values[index]) atIndex:2 + index];
  }
  const NSUInteger threads = std::min<NSUInteger>(
      impl_->istft_pipeline.maxTotalThreadsPerThreadgroup, 256);
  [encoder dispatchThreads:MTLSizeMake(original_samples, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [command_buffer commit];
  check_status(command_buffer, "metal::istft");
  copy_output(output_buffer, output);
  return output;
}

}  // namespace runtime
