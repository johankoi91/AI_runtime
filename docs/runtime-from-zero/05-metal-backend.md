# 第 5 课：CPU 怎样让 GPU 算一个结果

[上一课](04-graphs-and-onnx.md) · [下一课](06-hush-pipeline.md)

源码入口：[metal_backend.h](../../include/runtime/metal_backend.h)、[metal_backend.mm](../../src/backend/metal_backend.mm)、[metal_demo.cpp](../../apps/metal_demo.cpp)。

## 1. 一个文件里有两种执行环境

`metal_backend.mm` 前半部分的 `kMetalSource` 是一段 Metal Shading Language 字符串，包含在 GPU 上执行的 kernel。

后半部分是运行在 CPU 上的 Objective-C++：创建设备、编译 shader、申请 Buffer、绑定参数、提交命令、等待结束，再将结果交回 Tensor。

看到文件后缀是 `.mm`，不代表里面的每一行都在 GPU 上执行。

## 2. 六个 Metal 对象

| 对象 | 可以先这样理解 | 源码中的位置 |
| --- | --- | --- |
| `MTLDevice` | 当前可使用的 GPU 设备接口 | `Impl::device` |
| `MTLBuffer` | GPU 可访问的一段数据空间 | input/weight/output 缓存 |
| `MTLComputePipelineState` | 已准备好的某个计算程序 | `add_pipeline` 等 |
| `MTLCommandQueue` | 接收提交命令的队列 | `Impl::queue` |
| `MTLCommandBuffer` | 一次提交的命令包 | 各算子方法里的局部对象 |
| `MTLComputeCommandEncoder` | 把计算操作和参数写进命令包 | `setBuffer`、`dispatchThreads` |

MetalBackend 构造时编译 shader 并建立 pipeline。当前不是把编译好的 metallib 作为资源内置，也没有实现跨 App 启动的自有磁盘编译缓存。

## 3. 跟一遍 Add

在 `MetalBackend::add()` 中按顺序找这些动作：

```text
检查 dtype、shape、连续性
    → 从 CPU MemoryPool 分配输出 Tensor
    → 准备 lhs、rhs 的 Metal Buffer
    → 准备 Metal 输出 Buffer
    → 创建 command buffer 与 encoder
    → 选择 add_pipeline
    → 绑定 lhs、rhs、output、count
    → 安排线程
    → endEncoding
    → commit
    → waitUntilCompleted
    → 复制结果到输出 Tensor
```

注意这里同时存在“CPU 输出 Tensor”和“Metal 输出 Buffer”。它们在当前接口下不是同一个对象。

## 4. GPU 线程如何知道自己算哪个数字

`add_kernel` 的线程获得 `thread_position_in_grid`，代码把它叫作 index。

假设数组有四个元素，逻辑上安排 index 为 0、1、2、3 的线程，每个线程各算一个输出：

```text
thread 0：output[0] = left[0] + right[0]
thread 1：output[1] = left[1] + right[1]
thread 2：output[2] = left[2] + right[2]
thread 3：output[3] = left[3] + right[3]
```

MatMul kernel 使用二维线程坐标决定输出行列，再在该线程内累加 inner 维。当前仍是基础实现，没有实现成熟矩阵库那样的分块、共享存储与专用矩阵指令优化。

grid 是一共安排多少逻辑工作；threadgroup 是如何把线程分组。它们不是同一个数字，也不是 CPU 线程数量。

## 5. 统一内存为什么仍然有 memcpy

Apple Silicon 的 CPU/GPU 能共享物理内存，Metal Buffer 也使用 Shared storage，但这不会自动让普通 C++ Tensor 与 Buffer 指向同一地址。

当前 `make_input_buffer()` 将 Tensor 内容 memcpy 到 Buffer；`copy_output()` 将 Buffer 内容 memcpy 回 Tensor。是否复制，由数据结构与执行接口的设计决定。

因此，“统一内存硬件”与“本 runtime 已经零拷贝”是两回事。

## 6. 当前缓存复用了什么

| 缓存 | key | 用途 |
| --- | --- | --- |
| `input_buffers` | 大小与用途 slot 组合 | 复用输入 Buffer，内容每次更新 |
| `weight_buffers` | 权重内容 hash | 复用相同权重的 Metal Buffer |
| `output_buffers` | 输出字节数 | 复用临时 Metal 输出空间 |
| pipeline 字段 | 每个算子一个已建对象 | 复用编译结果 |

权重缓存命中可以省下重复创建和上传，但当前仍可能重复计算内容 hash。缓存按出现过的大小/内容保留，也不是带容量预算的通用缓存管理器。

复用输出空间之所以适用于当前顺序调用，是因为每次等待计算结束并复制输出后才继续。若改成多个算子并行，需要先设计 Buffer 生命周期，防止结果还未读走就被覆盖。

## 7. 已经存在的同步开销

`check_status()` 中明确调用 `[command_buffer waitUntilCompleted]`。各算子还会创建 command buffer/encoder。

所以当前实现不能描述为“整张图共用一个 encoder”“已经消除逐节点等待”或“所有中间 Tensor 常驻 GPU”。这些是可以继续做的优化方向。

优化的依赖关系是：先知道每个中间结果的使用者与寿命，再安排驻留和复用，接着减少提交/复制/等待，同时用数值与内存测试检查是否破坏正确性。

## 8. GRU 的真实执行位置

- `MetalBackend::gru_step()` 一开始就返回 `cpu::gru_step()`，后面的 GPU 编码代码当前不可达。
- `MetalBackend::gru_sequence()` 默认使用 CPUBackend；实验环境变量可以选择另一条实现，但不能因此认为它是已验收的通用 GRU 路径。
- Hush ONNX 的 GRU 由 OnnxExecutor 内的实现执行，不走以上 GRU shader。
- Hush 的前端和合成也有自己的 CPU 实现。

Metal 在本项目中有实际执行的算子，并不是界面上的装饰；同时，Hush 当前是 CPU 与 Metal 混合执行。

## 9. 自测与实验

运行：

```sh
./build-app/bin/mini_runtime_metal_demo
```

如果机器没有可用 Metal 设备，这个目标不能证明 GPU 路径。macOS 上还可以对照日志，找到 `kernel_submitted`、`kernel_completed`、`copy_back_completed`。

回答三个问题：

1. `commit()` 返回时是否保证 GPU 已完成？
2. 为什么删除 wait 后立即 memcpy 会有问题？
3. 为什么把多个调用并发起来之前，要重新审查缓存的输出 Buffer？

答案：不保证；CPU 可能读到尚未完成的结果；两个调用可能同时写相同缓存空间。去掉等待是一项执行架构变更，不能只删除一行代码。
