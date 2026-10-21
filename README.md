# mini-llm-runtime

一个从零实现的最小 CPU 推理 Runtime，用来学习 runtime 最核心的几层能力：

- Tensor 存储和 layout
- 基础 CPU kernel
- Memory Pool
- Backend 抽象
- Execution Graph
- 后续扩展到 KV Cache 和最小 decoder path

## 当前状态

当前仓库已经不是空骨架，而是一个能本地编译、运行和测试的最小闭环。

目前已经实现：

- Hush 接入阶段 1：
  - 16 kHz 单声道流式音频帧和 10 ms 分帧队列
  - 带 shape/name/value 校验的 float32 权重文件基础格式
- Hush 接入阶段 2：
  - 320 点 FFT、160 点 hop 的复数 STFT
  - Hann 窗口和重叠相加 ISTFT
  - CPU 参考实现，供后续 Metal 算子逐点对齐
- Hush 接入阶段 3：
  - Metal STFT kernel
  - CPU/Metal 频谱逐元素误差验证
- Hush 接入阶段 4：
  - CPU/Metal sigmoid 和 tanh 激活
  - 激活算子 CPU/Metal 数值对齐
- Hush 接入阶段 5：
  - 单步 GRU CPU 参考实现
  - Metal GRU kernel 和状态输入输出
  - CPU/Metal GRU 数值对齐
- Hush 接入阶段 6：
  - NCHW Conv2D CPU 参考实现
  - Metal Conv2D kernel，支持 stride、padding 和 bias
  - CPU/Metal Conv2D 数值对齐
- Hush 接入阶段 7：
  - NCHW ConvTranspose2D CPU 参考实现
  - Metal ConvTranspose2D kernel，使用 PyTorch 常见权重布局
  - CPU/Metal ConvTranspose2D 数值对齐
- Hush 接入阶段 8：
  - ERB filterbank 能量投影和 log 特征
  - ExponentialUnitNorm 运行能量状态
  - CPU/Metal ERB 与归一化数值对齐

- `Tensor`
  - `float32`
  - shape、stride、offset
  - contiguous layout
  - `view`
- CPU kernel
  - `add`
  - naive `matmul`
- `MemoryPool`
  - 简单 arena 风格分配
  - `reset / reuse`
  - 基础内存统计
- `CPUBackend`
  - 通过 backend 接口调用 `add`、`matmul`
- `Graph + Executor`
  - 最小图执行路径
  - 当前 demo 跑的是 `matmul -> add`
- `tests`
  - Tensor、kernel、memory pool、graph execution 的基础验证

## 本地运行

### 构建

```bash
cmake -S . -B build
cmake --build build
```

### 运行 demo

```bash
./build/bin/mini_runtime_demo
```

### 运行测试

```bash
ctest --test-dir build --output-on-failure
```

### 在 macOS 上运行 Metal GPU demo

Apple Silicon 上可以用 Metal backend 执行当前支持的 `add` 和 `matmul`：

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --parallel
./build-metal/bin/mini_runtime_metal_demo
```

这个 demo 会执行 `matmul -> add` 图，并校验 GPU 输出为 `[32, 48, 79, 104]`。

### 使用远端 InfluxDB 记录真实运行事件

远端部署完成后，本地 M2 demo 可以把 Metal 事件写入远端：

```bash
INFLUXDB_URL=http://81.71.150.26/influx \
INFLUXDB_ORG=mini-runtime \
INFLUXDB_BUCKET=mini-runtime \
INFLUXDB_TOKEN=mini-runtime-local-token \
./build-metal/bin/mini_runtime_metal_demo
```

页面地址：`http://81.71.150.26/`。页面查询最近 24 小时事件，并按最近一次 `run_id` 的真实时间戳回放。

Metal trace 现在还会记录设备和 pipeline 就绪、Node ready、输入读取、MTLBuffer 创建、command buffer、encoder、dispatch grid、kernel 提交/完成，以及结果回传开始/完成等细粒度事件。事件先在本地批量缓存，运行结束后一次写入 InfluxDB，避免网络请求干扰 GPU 执行时间。

### 打开动画讲解界面

如果想直观看每一步怎么流转，可以直接打开 `apps/metal_visualizer.html`，点击“从头运行”。
它会像一个小型 GPU profiler 一样展示代码高亮、Tensor shape/stride、Host buffer、Metal shared buffer、GPU 线程格子、矩阵单元格计算和事件日志；可以暂停、单步和调速。

## 目录结构

```text
include/runtime/        核心头文件
src/                    核心实现
src/backend/            backend 实现
src/kernels/cpu/        CPU kernel
apps/demo.cpp           最小可运行示例
tests/runtime_tests.cpp 轻量测试
docs/                   学习路线和任务文档
```

## 学习文档

- [Runtime 学习路线图](./docs/runtime-learning-roadmap.md)
- [第 1 周任务清单](./docs/week-1-task-checklist.md)
- [源码阅读 Guide](./docs/source-reading-guide.md)

路线图已经按当前仓库状态对齐过了。  
也就是说，仓库里已经提前有了一部分第 2-3 周的内容，而第 1 周更适合做“补齐缺口、加深理解、增强测试”。

## 当前最值得做的下一步

如果你按路线继续推进，最建议优先做这些：

1. 补 `mul`
2. 补强 `Tensor` 的 `view / stride / offset` 测试
3. 写 `docs/tensor-layout-notes.md`
4. 给 `matmul` 加更完整的错误路径测试
5. 再进入 blocked `matmul`、benchmark、KV Cache

## Hush Metal 移植阶段自测

每个移植阶段完成后，先运行这一组检查：

```bash
cmake -S . -B build-stage -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-stage --parallel
ctest --test-dir build-stage --output-on-failure
./build-stage/bin/mini_runtime_metal_demo
git diff --check
```

阶段 1 额外覆盖音频帧边界、音频格式、权重文件 round-trip 和权重元数据校验。阶段 2 增加 STFT/ISTFT 的 bin 数、采样数和有效重叠区 round-trip 检查。阶段 3 增加 CPU/Metal STFT 最大误差检查，本机验证值为 `0.000154613`。阶段 4 增加 CPU/Metal sigmoid/tanh 对齐检查，本机最大误差为 `1.19e-7`。阶段 5 增加单步 GRU 对齐检查，本机最大误差为 `4.47e-8`。阶段 6 增加 Conv2D 对齐检查，本机最大误差为 `0`。阶段 7 增加 ConvTranspose2D 对齐检查，本机最大误差为 `0`。阶段 8 增加 ERB 投影和 ExponentialUnitNorm 对齐检查，最大误差分别为 `1.19e-7` 和 `0`。后续阶段会在这组基础检查上增加模型算子和 Hush 参考输出对齐测试。

## 当前能力边界

当前版本故意保持最小化，所以也有一些明确限制：

- 目前主要围绕 `float32`
- kernel 只支持很小的一组能力
- `matmul` 目前只支持 contiguous 的 2D tensor
- 还没有 `mul`、`softmax`、`rmsnorm`
- 还没有 `KernelRegistry`
- 还没有 `KVCache`

这些限制不是问题，反而正适合拿来当学习入口。
