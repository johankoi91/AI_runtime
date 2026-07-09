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

## 当前能力边界

当前版本故意保持最小化，所以也有一些明确限制：

- 目前主要围绕 `float32`
- kernel 只支持很小的一组能力
- `matmul` 目前只支持 contiguous 的 2D tensor
- 还没有 `mul`、`softmax`、`rmsnorm`
- 还没有 `KernelRegistry`
- 还没有 `KVCache`

这些限制不是问题，反而正适合拿来当学习入口。
