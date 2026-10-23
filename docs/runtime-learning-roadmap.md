# Runtime 学习路线图

> 本文保留最初的规划，包含 Softmax、RoPE、KV Cache 等尚未实现的学习方向。阅读当前已实现的 CPU/Metal、ONNX、Hush 与 App，请使用 [从 0 到 1 源码课程](runtime-from-zero/README.md)。

## 目标

这份文档的目标不是教你“怎么调一个现成框架”，而是帮你从零理解并实现一个最小但完整的 Runtime。

学习重点放在下面这些能力：

- Tensor 是怎么存储的
- MatMul 和基础算子怎么实现
- 算子调度怎么设计
- Memory Pool 怎么减少频繁分配
- KV Cache 怎么组织和复用
- Execution Graph 怎么表示和执行
- CPU Backend 怎么抽象

这条路线默认以 CPU 为主。等这些核心概念跑通后，再迁移到 CUDA/GPU 会顺很多。

## 你现有设备怎么用

- `Mac M2 16GB`
  - 写代码、跑小规模实验、做可视化和文档整理
  - 适合开发一个 CPU 版 mini runtime
- `16c 32G Linux 云服务器`
  - 跑 Linux 环境 benchmark
  - 做服务化、压测、性能对比
  - 验证跨平台构建和脚本

结论：你现在完全可以开始，不需要先买 GPU。

## 学习原则

- 先追求正确性，再追求性能
- 先做最小闭环，再做泛化和抽象
- 每个阶段都要有 benchmark 和测试
- 每做完一层，就写一页设计笔记

## 能力地图

### 1. Tensor

你需要真正吃透的不是“数组”，而是下面这些概念：

- `shape`
- `stride`
- `dtype`
- `storage`
- `offset`
- contiguous 和 non-contiguous view

学到这里，你应该能回答：

- 为什么转置通常不复制数据
- 为什么有些 view 可以零拷贝
- 为什么某些 kernel 只想吃 contiguous tensor

### 2. Kernel

先实现最基本的 CPU kernel：

- `add`
- `mul`
- `matmul`
- `softmax`
- `rmsnorm`
- `rope`

关键不是算子数量，而是你要意识到：

- kernel 有输入输出约束
- kernel 会依赖 tensor layout
- kernel 的性能会被 cache locality 影响

### 3. Dispatcher / Scheduler

你至少要抽象出：

- op definition
- kernel registry
- backend dispatch
- execution order

这一层的目标是：同一个 `matmul`，未来可以挂不同实现。

### 4. Memory Pool

你要理解 Runtime 为什么不喜欢频繁 `malloc/free`：

- 分配慢
- 容易碎片化
- 生命周期难管理

最小版只需要先做：

- arena 或 slab 风格分配
- reset / reuse
- 简单的统计指标

### 5. KV Cache

这部分是你未来做 LLM Runtime 的关键桥梁。

你至少要理解：

- 为什么 attention 会重复读历史 token
- K/V 为什么要缓存
- cache layout 怎么影响访问效率
- append-only 和 paged cache 的区别

### 6. Execution Graph

你需要一个图来表达：

- 节点是什么
- 边是什么
- 输入输出怎么绑定
- 执行顺序怎么确定

最小实现只要支持：

- DAG
- 拓扑排序
- 顺序执行

### 7. Backend

Backend 的核心不是“换个平台”，而是把：

- 张量表示
- kernel 实现
- 分配方式
- 调度入口

组织成一个可替换的执行后端。

先做 `CPUBackend` 就够了。

## 推荐技术栈

为了把重点放在 Runtime 本身，我建议：

- 语言：`C++`
- 构建：`CMake`
- 测试：`GoogleTest` 或先用简单断言
- benchmark：`Google Benchmark` 或自写计时器
- BLAS 对照：`OpenBLAS` 或 `oneDNN`

如果你想先快速验证设计，也可以先写一版 `Python + NumPy` 原型，再落成 `C++`。

但如果目标是 Runtime，本路线更推荐直接上 `C++`。

## 4 周路线

这个 4 周版本不是删内容，而是把原来 12 周的内容压成一个更高密度的节奏。

更准确地说，它是：

- 每周都有明确交付物
- 每周都同时做实现、测试、文档、benchmark
- 允许你在一周内并行推进 2-3 个相关主题

如果你每天能投入 `2-4` 小时，这个节奏是可行的。  
如果你投入更少，可以继续按这 4 周结构推进，但把每周拆成两周。

### 第 1 周：Tensor、基础算子、MatMul 起步

这一周的重点是先把 Runtime 最底层打稳。你要同时完成张量表示和最基本的 kernel，这样后面的 graph、pool、cache 才不会漂。

目标：

- 实现最小 `Tensor`
- 支持 `float32`
- 支持 shape、stride、offset
- 支持 contiguous tensor
- 理解 view 和 layout 的关系
- 实现 `add`、`mul`
- 实现 naive `matmul`

产出：

- `Tensor` 类
- 基础单元测试
- `add`、`mul`、naive `matmul`
- 一页文档：tensor layout 设计说明

验收标准：

- 能创建 1D/2D tensor
- 能做 elementwise add/mul
- 能正确打印 shape/stride
- 矩阵乘法结果正确

本周建议安排：

- 前 2 天：完成 `Tensor`、`Storage`、`DType`
- 第 3 天：完成 `add`、`mul`
- 第 4-5 天：完成 naive `matmul`
- 第 6-7 天：补测试，写 tensor layout 设计笔记

### 第 2 周：MatMul 优化、Memory Pool、生命周期管理

这一周的重点是从“能跑”进入“有 Runtime 味道”。你要开始接触性能和内存管理，而不是只停留在算子正确性。

目标：

- 增加 blocked matmul
- 对比 `OpenBLAS` 或 `oneDNN`
- 设计一个最小 memory pool
- Tensor 改为从 pool 分配
- 记录已分配字节数、峰值占用、复用次数

产出：

- 两版 matmul kernel
- benchmark 脚本
- `MemoryPool`
- 简单统计接口
- 文档：为什么 blocking 会更快
- 文档：为什么 Runtime 需要内存池

验收标准：

- 不同矩阵规模下能看到 naive 和 blocked 的性能差异
- 重复执行相同计算路径时分配次数明显下降
- 能输出一份内存使用摘要

本周建议安排：

- 前 2 天：实现 blocked `matmul`
- 第 3 天：补 benchmark，对比 naive 和 blocked
- 第 4-5 天：实现 `MemoryPool`
- 第 6-7 天：让 tensor allocation 接入 pool，补统计和设计文档

### 第 3 周：Execution Graph、Dispatcher、CPU Backend

这一周的重点是把零散算子升级成真正的执行系统。做到这一步，你的项目才开始像一个 runtime，而不是一堆 kernel 文件。

目标：

- 抽象 op node
- 建图
- 拓扑排序
- 顺序执行
- backend dispatch
- 定义 `CPUBackend`
- 设计 `KernelRegistry`

产出：

- `Op`
- `Graph`
- `Executor`
- `KernelRegistry`
- `CPUBackend`
- 一个可执行的 `matmul -> add -> rmsnorm` 或 `matmul -> add` 小图

验收标准：

- 能构建计算图并正确执行
- executor 不直接依赖某个具体 kernel 实现
- 新增一个 kernel 不需要改 executor 主逻辑

本周建议安排：

- 前 2 天：定义 graph/node/op 抽象
- 第 3-4 天：完成 executor 和顺序执行
- 第 5 天：接入 backend dispatch
- 第 6-7 天：整理接口，补图执行测试

### 第 4 周：KV Cache、最小 Attention 路径、收尾和总结

这一周的重点是把 Runtime 拉到更接近 LLM 推理的语境里。你不需要做完整模型，但要把 KV Cache 和最小 decoder step 串起来。

目标：

- 实现最小 attention 所需的 K/V 缓存
- 支持 token-by-token append
- 支持读取历史 cache
- 实现一个极简 decoder step 演示
- 统一接口
- 补测试
- 补 benchmark
- 输出项目总结

产出：

- `KVCache`
- 一个极简 decoder step 演示
- 文档：prefill 和 decode 的差异
- README
- benchmark 报告
- 架构图
- 后续 GPU 迁移计划

验收标准：

- 多轮 append 后历史 K/V 可正确读取
- 能解释 cache layout 对性能的影响
- 项目可以从头构建运行
- 关键模块都有最小测试
- 你能用 10 分钟讲清整个系统

本周建议安排：

- 前 2 天：设计并实现 `KVCache`
- 第 3-4 天：补一个最小 decoder step
- 第 5 天：补 KV cache benchmark 和测试
- 第 6-7 天：整理 README、架构图和总结文档

## 当前仓库对齐状态

这份路线图现在已经和当前仓库的代码状态对过一遍。结论是：当前项目并不是“从第 1 周起步的空仓库”，而是已经提前实现了一部分第 2-3 周的内容。

当前仓库里已经有：

- `Tensor` 的最小实现
- `float32`、shape、stride、offset、`view`
- `add`
- naive `matmul`
- `MemoryPool`
- `CPUBackend`
- `Graph + Executor`
- demo 和基础测试

对应文件主要是：

- `include/runtime/tensor.h`
- `src/tensor.cpp`
- `src/kernels/cpu/add.cpp`
- `src/kernels/cpu/matmul.cpp`
- `src/memory_pool.cpp`
- `include/runtime/cpu_backend.h`
- `src/backend/cpu_backend.cpp`
- `include/runtime/graph.h`
- `src/graph.cpp`
- `src/executor.cpp`
- `tests/runtime_tests.cpp`

这意味着你现在做第 1 周时，不需要从零开始，而是应该采用“补齐缺口、加深理解、增强测试”的方式推进。

当前和第 1 周目标相比，最值得补的缺口是：

- `mul` 还没有实现
- `view / stride / offset` 的测试覆盖还比较薄
- non-contiguous layout 目前基本只停留在表示层，还没有充分验证
- 还缺少一份专门解释 tensor layout 的设计笔记

为了避免主文档太长，我把“第 1 周任务清单”单独整理成了一份更贴近当前代码的执行清单：

- [第 1 周任务清单](./week-1-task-checklist.md)

## 下一步建议

最实用的起点不是继续看一堆资料，而是今天就开一个仓库，然后做下面三件事：

1. 建 `Tensor` 和 `DType`
2. 写 `add` 和 `matmul`
3. 补第一批测试

等这三件事跑通，你的学习就已经从“理解概念”进入“真正做 Runtime”了。
