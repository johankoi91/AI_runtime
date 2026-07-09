# 第 1 周任务清单

这份清单不是按“理想空仓库”写的，而是按你当前仓库的实际状态写的。

换句话说，下面的任务不是让你重做已经有的代码，而是：

- 补齐第 1 周还没完成的部分
- 把已经有的代码理解得更扎实
- 让仓库状态和路线图的第 1 周目标真正对齐

## 当前状态

### 已经完成

- `Tensor` 的最小表示已经有了
- `float32`、shape、stride、offset 都已经有基础实现
- `add` 已经实现
- naive `matmul` 已经实现
- demo 和基础测试已经能本地跑通

### 超前完成

- `MemoryPool`
- `CPUBackend`
- `Graph`
- `Executor`

这些内容本来更接近第 2-3 周，但仓库里已经提前存在了。

### 还没补齐

- `mul` 还没有实现
- `Tensor view` 的语义和测试还不够扎实
- 缺一份专门的 tensor layout 设计笔记
- 第 1 周层面的测试还可以更细

## 第 1 周目标

这一周你真正要完成的是下面 4 件事：

1. 把 `Tensor` 这层补结实
2. 把 `mul` 补上
3. 把 `naive matmul` 的理解和测试补完整
4. 留下一份你自己能复盘的 tensor layout 文档

## 任务清单

### 任务 1：补强 Tensor 的 layout 语义

目标：

- 让 `shape / stride / offset / view` 的语义更清晰
- 让 `Tensor` 不只是“能存数据”，而是真的能支撑后面的 runtime 抽象

建议改这些文件：

- `include/runtime/tensor.h`
- `src/tensor.cpp`
- `tests/runtime_tests.cpp`

建议补的内容：

- 给 `Tensor` 增加更明确的 layout 辅助接口
  - 比如 `is_scalar`、`is_empty`、`dim`、`stride_at`
  - 或一个根据多维索引计算元素偏移的小工具
- 检查 `view` 和 `offset` 的边界是否合法
- 重新审视 `validate()` 是否真的覆盖了 stride 带来的跨度问题

这里最值得你自己想明白的一点是：

- 当前 `validate()` 主要是按 `numel()` 去估算 buffer 是否够用
- 但一旦 stride 不是 contiguous，真实访问跨度可能大于 `numel()`

这就是一个很典型的 runtime 设计问题，很适合作为你第 1 周的重点思考题。

完成标准：

- 你能解释 contiguous 和 non-contiguous 的差别
- 你能用测试覆盖 `view + offset + stride`
- 你能说清现在的 `Tensor` 还支持什么，不支持什么

### 任务 2：实现 `mul`

目标：

- 补齐第 1 周计划里的基础 elementwise kernel

建议改这些文件：

- `include/runtime/kernels.h`
- `src/kernels/cpu/mul.cpp`
- `CMakeLists.txt`
- `tests/runtime_tests.cpp`

如果你想让当前架构更一致，可以额外改：

- `include/runtime/backend.h`
- `include/runtime/cpu_backend.h`
- `src/backend/cpu_backend.cpp`
- `include/runtime/graph.h`
- `src/executor.cpp`

但这部分是加分项，不是第 1 周必做项。

最低完成标准：

- `runtime::cpu::mul(lhs, rhs)` 能正确工作
- 支持和 `add` 一样的 shape 校验
- 有对应测试

更完整的完成标准：

- `mul` 也能通过 `Backend` 路径调用
- 如果你愿意，再把 `OpType::Mul` 接进图执行

### 任务 3：把 `matmul` 当成第 1 周重点算子来复盘

目标：

- 不只是“代码有了”，而是你自己要彻底看懂 naive `matmul`

建议改这些文件：

- `src/kernels/cpu/matmul.cpp`
- `tests/runtime_tests.cpp`
- 可选：`apps/demo.cpp`

建议补的内容：

- 给 `matmul` 加更细的异常测试
  - rank 不对
  - inner dimension 不匹配
  - 非 contiguous 输入
- 给 `matmul` 补几组更小和更直观的数据样例
- 如果你愿意，给 `demo` 增加打印，帮助你肉眼追踪矩阵结果

完成标准：

- 你能手算出 demo 里的矩阵乘法结果
- 你能解释三层循环每一层在干什么
- 你能解释为什么当前实现只支持 contiguous tensor

### 任务 4：补一份 tensor layout 设计笔记

目标：

- 把“我看懂了”变成“我能写清楚”

建议新增文件：

- `docs/tensor-layout-notes.md`

建议内容：

- `shape` 是什么
- `stride` 是什么
- `offset` 是什么
- contiguous tensor 怎么判断
- 为什么 transpose 常常不需要复制
- 为什么有些 kernel 暂时只吃 contiguous 输入
- 你当前实现里支持和不支持的边界

完成标准：

- 你自己一周后回来看这份笔记，仍然能快速想起实现思路
- 这份笔记能和 `Tensor` 代码一一对应起来

### 任务 5：把测试从“有”升级到“能说明问题”

目标：

- 让测试不只是验证 happy path，而是帮助你理解 runtime 边界

建议改这些文件：

- `tests/runtime_tests.cpp`

建议补的测试：

- `from_vector` 输入数量不匹配时报错
- `view` 的 offset 是否正确
- 非法 stride / shape 组合是否被拦住
- `mul` 的正确性
- `matmul` 的错误路径

完成标准：

- 测试既覆盖正确性，也覆盖失败路径
- 你能从测试名字看出当前 runtime 的能力边界

## 建议顺序

如果你按 7 天推进，我建议这样排：

### Day 1

- 通读 `Tensor` 相关代码
- 在 `tests/runtime_tests.cpp` 里先补 `Tensor` 和 `view` 测试草稿

重点文件：

- `include/runtime/tensor.h`
- `src/tensor.cpp`
- `tests/runtime_tests.cpp`

### Day 2

- 修 `Tensor::validate()`
- 明确 stride / offset 的边界

重点文件：

- `src/tensor.cpp`
- `tests/runtime_tests.cpp`

### Day 3

- 实现 `mul`
- 先不接 graph，先保证 kernel 正确

重点文件：

- `include/runtime/kernels.h`
- `src/kernels/cpu/mul.cpp`
- `CMakeLists.txt`
- `tests/runtime_tests.cpp`

### Day 4

- 复盘 `matmul`
- 补 `matmul` 错误路径测试

重点文件：

- `src/kernels/cpu/matmul.cpp`
- `tests/runtime_tests.cpp`

### Day 5

- 决定要不要把 `mul` 接进 `Backend`
- 如果接，就顺着接口一路打通

重点文件：

- `include/runtime/backend.h`
- `include/runtime/cpu_backend.h`
- `src/backend/cpu_backend.cpp`
- `include/runtime/graph.h`
- `src/executor.cpp`

### Day 6

- 写 `docs/tensor-layout-notes.md`
- 对照代码补文档里的边界说明

重点文件：

- `docs/tensor-layout-notes.md`
- `include/runtime/tensor.h`
- `src/tensor.cpp`

### Day 7

- 整体回归
- 跑构建、测试、demo
- 写一段本周总结

重点文件：

- `README.md`
- `tests/runtime_tests.cpp`
- `apps/demo.cpp`

## 本周结束时你应该达到的状态

到第 1 周结束时，最理想的状态不是“功能很多”，而是：

- 你真的理解了 `Tensor` 的 layout
- 你补齐了 `mul`
- 你能把 naive `matmul` 讲明白
- 你有测试证明自己的实现边界
- 你有一份自己的 tensor 设计笔记

如果这几件事都完成了，你的第 1 周就是扎实的。后面进入 memory pool、graph、KV cache 时会轻松很多。
