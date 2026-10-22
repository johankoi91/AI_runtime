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
- Hush 接入阶段 9：
  - ONNX ModelProto/GraphProto/TensorProto initializer 读取
  - 支持 float32 shape、name、raw_data 和 float_data
  - 损坏 protobuf、非 float32 和 shape 不匹配会明确失败
- Hush 接入阶段 10：
  - gzip/tar Hush 模型 bundle 读取
  - 支持列出和提取 `.onnx`、`config.ini` 等普通文件条目
  - tar 边界、gzip 损坏和截断文件检测
- Hush 接入阶段 11：
  - ONNX graph name 和 node 顺序读取
  - NodeProto 的 op type、输入和输出名称读取
  - graph initializer 与节点元数据统一返回
- Hush 接入阶段 12：
  - `HushModel` 聚合配置和三张 ONNX 子图
  - 统一加载 `enc.onnx`、`erb_dec.onnx`、`df_dec.onnx`
  - 验证子图节点类型和 bundle 条目映射
- Hush 接入阶段 13：
  - 解析 ONNX `AttributeProto`
  - 支持 Conv/GRU 所需整数和整数数组属性
  - 支持按名称查找 initializer，准备 graph 到 kernel 的映射
- Hush 接入阶段 14：
  - ONNX 节点执行器
  - `Add`、`MatMul` 映射到统一 Backend
  - CPU backend 执行 graph fixture
- Hush 接入阶段 15：
  - ONNX executor 接入 `Conv`、`Relu`、`Sigmoid`、`Tanh`
  - 接入受限的单步单方向 `GRU`
  - CPU 和 Metal backend 统一算子分发
- Hush 接入阶段 16：
  - ONNX executor 接入 `Reshape`、`Transpose`、`Concat`、`Flatten`
  - 增加 shape kernel 的连续布局和维度校验
  - 用 CPU 参考路径验证 Hush graph 的连接层变换
- Hush 接入阶段 17：
  - `HushStreamSession` 流式 encoder 执行入口
  - hidden state 的首帧注入、后续帧自动回灌和 reset
  - frame counter 与 state binding 自测
- Hush 接入阶段 19：
  - 解析 ONNX graph 的正式输入和输出名称
  - Hush session 执行前校验必需 encoder 输入
  - 用 fixture 验证输入输出元数据和缺失输入拒绝
- Hush 接入阶段 20：
  - 解析 Hush `config.ini` 的 `[df]` 参数
  - 以模型配置约束 sample rate、FFT size 和 hop size
  - 非法配置和错误音频帧长度拒绝
- Hush 接入阶段 21：
  - Hush graph inventory 报告节点数量和 op 类型
  - 标记当前 executor 尚未支持的 op
  - 为真实模型接入前的算子覆盖检查提供入口
- Hush 接入阶段 22：
  - 解析 Conv 的 `strides`、`pads`、`dilations` 和 `group`
  - 支持二维对称 padding 和 stride
  - 对 dilation、group 和非对称 padding 做明确拒绝
- Hush 接入阶段 23：
  - ONNX executor 接入 `ConvTranspose`
  - CPU/Metal backend 统一反卷积分发
  - 修复 MemoryPool 输出复用时反卷积累加缓冲未清零问题
- Hush 接入阶段 24：
  - 校验 Hush 三张子图非空
  - 校验 graph 正式输入输出存在
  - 校验模型只使用当前支持的 op 类型
- Hush 接入阶段 25：
  - 16 kHz/160 samples 流式前端
  - 320 点 STFT 频谱 Tensor
  - 32 band ERB 特征 Tensor
- Hush 接入阶段 26：
  - STFT/ERB 接入统一 `Backend` 接口
  - CPU backend 提供参考实现
  - Metal backend 直接执行前端 kernel
- Hush 接入阶段 27：
  - 截取最低 64 个频率 bin
  - 生成双通道 real/imag DF 特征
  - 对 DF 特征做逐帧能量归一化
- Hush 接入阶段 28：
  - `HushStreamSession` 接入 `HushFrontend`
  - 音频帧到 encoder feature input 的绑定入口
  - session reset 同步清理 frontend 状态
- Hush 接入阶段 29：
  - 逐元素 `Mul` CPU/Metal kernel
  - 统一 Backend 和 ONNX executor 分发
  - 严格 matching shape 校验
- Hush 接入阶段 30：
  - 音频帧到 frontend、encoder 的端到端 orchestration
  - ERB/DF 双特征输入绑定
  - session 状态和前端状态统一推进
- Hush 接入阶段 18：
  - `AudioFrame` 接入流式 encoder session
  - 强制 Hush 的 16 kHz、单声道、160 samples/frame 契约
  - encoder state output 图约束和非法音频帧拒绝

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

阶段 1 额外覆盖音频帧边界、音频格式、权重文件 round-trip 和权重元数据校验。阶段 2 增加 STFT/ISTFT 的 bin 数、采样数和有效重叠区 round-trip 检查。阶段 3 增加 CPU/Metal STFT 最大误差检查，本机验证值为 `0.000154613`。阶段 4 增加 CPU/Metal sigmoid/tanh 对齐检查，本机最大误差为 `1.19e-7`。阶段 5 增加单步 GRU 对齐检查，本机最大误差为 `4.47e-8`。阶段 6 增加 Conv2D 对齐检查，本机最大误差为 `0`。阶段 7 增加 ConvTranspose2D 对齐检查，本机最大误差为 `0`。阶段 8 增加 ERB 投影和 ExponentialUnitNorm 对齐检查，最大误差分别为 `1.19e-7` 和 `0`。阶段 9 使用最小 ONNX fixture 验证 initializer 读取。阶段 10 使用最小 tar.gz fixture 验证 bundle 解压和条目读取。阶段 11 使用 ONNX fixture 验证 graph 和 node 元数据读取。阶段 12 验证 Hush 三张子图和 config 的聚合加载。阶段 13 使用 ONNX fixture 验证 AttributeProto 和按名权重映射。阶段 14 验证 ONNX `Add` 节点通过统一执行器调用 backend。阶段 15 验证 Conv、激活和单步 GRU 节点分发。阶段 16 验证 Reshape、Transpose、Concat 和 Flatten 的连接层变换。阶段 17 验证 state binding 在多帧 encoder 执行中的注入、回灌和 reset。阶段 18 验证 16 kHz 单声道 160 samples/frame 契约和非法音频帧拒绝。阶段 19 验证 ONNX graph 输入输出名称读取和 session 缺失输入拒绝。阶段 20 验证 Hush config 参数解析、模型音频契约和错误帧拒绝。阶段 21 验证 graph inventory 和当前 op 覆盖报告。阶段 22 验证 Conv 属性解析和受限二维参数校验。阶段 23 验证 ConvTranspose executor 分发及 MemoryPool 输出初始化。阶段 24 验证 Hush 三张子图的模型级 graph 契约。阶段 25 验证音频帧到 STFT/ERB 特征的前端链路。阶段 26 验证 STFT/ERB 通过统一 Backend 接口执行。阶段 27 验证最低 64 bin 的双通道 DF 特征和能量归一化。阶段 28 验证音频帧经 frontend 注入 encoder 以及 frontend reset。阶段 29 验证 Mul 逐元素算子和 ONNX 分发。阶段 30 验证音频帧到 frontend、ERB/DF 输入和 encoder 的端到端 orchestration。下一阶段进入 decoder 串联和增强频谱输出。

## 当前能力边界

当前版本故意保持最小化，所以也有一些明确限制：

- 目前主要围绕 `float32`
- kernel 只支持很小的一组能力
- `matmul` 目前只支持 contiguous 的 2D tensor
- 还没有 `mul`、`softmax`、`rmsnorm`
- 还没有 `KernelRegistry`
- 还没有 `KVCache`

这些限制不是问题，反而正适合拿来当学习入口。
