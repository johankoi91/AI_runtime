# 第 4 课：模型文件怎样变成函数调用

[上一课](03-cpu-kernels.md) · [下一课](05-metal-backend.md)

源码入口：[graph.h](../../include/runtime/graph.h)、[onnx_weights.cpp](../../src/onnx_weights.cpp)、[onnx_plan.cpp](../../src/onnx_plan.cpp)、[onnx_executor.cpp](../../src/onnx_executor.cpp)。

## 1. 先分清两套图

| | 最小手写图 | ONNX 模型图 |
| --- | --- | --- |
| 数据类型 | `Graph`、`Node`、`OpType` | `OnnxGraph`、`OnnxNode` |
| 算子表示 | Add、MatMul 枚举 | `"Conv"`、`"GRU"` 等字符串 |
| 主要应用 | `apps/demo.cpp` | Hush 和 ONNX 测试 |
| 执行入口 | `Executor::run()` | `OnnxExecutor::run()` |
| 顺序 | 按添加顺序执行 | 通过 ExecutionPlan 拓扑排序 |

两个执行器都使用 Backend/Tensor/MemoryPool，但 ONNX 不是先转换成仅有两个算子的最小 Graph。

## 2. 一个模型包括哪些内容

当前 `OnnxGraph` 保存：节点列表、输入名字、输出名字、initializer 权重，以及节点属性。

比如一个 Conv 节点需要输入名、权重名、可选 bias 名，还可能带 stride、padding、groups。只知道 op 名为 Conv，信息还不够。

权重和输入都用 Tensor 参与计算，区别是生命周期和来源：权重来自模型文件，输入来自这次调用，输出由算子计算得到。

## 3. 读取器先处理字节

ONNX 使用 Protobuf 二进制格式。[onnx_weights.cpp](../../src/onnx_weights.cpp) 中的 Reader 负责读取 varint、固定宽度数值和带长度的字段，并跳过暂时不处理的字段。

```text
文件字节
  → Reader 读取字段
  → parse_tensor / parse_node / parse_graph
  → WeightTensor / OnnxNode / OnnxGraph
```

这一步得到的是 C++ 数据结构，还没有执行模型。代码自己读取所需字段，没有把 ONNX Runtime 库嵌入 App。

当前运行时 Tensor 的 dtype 仍以 Float32 为主。加载时将部分整数或半精度数据转换成 float 保存，不等于具有原生 Int64、Float16 Tensor。某些 shape/index 值也按浮点数保存，具有精度与语义边界。

另有 [weights.cpp](../../src/weights.cpp) 实现简易的自定义权重文件读写，用于入门和测试；Hush 主路径使用 ONNX bundle。

## 4. 拓扑排序解决什么问题

假设文件中两项计算的存放顺序是：

```text
第二项：out = hidden + bias
第一项：hidden = a @ b
```

如果机械地先执行 Add，`hidden` 还不存在。`OnnxExecutionPlan::compile()` 先分析每个值由哪个节点产生，再计算每个节点还有多少依赖没有满足。

1. 把没有前置依赖的节点放入 ready 队列。
2. 从队列取一个节点，追加到执行顺序。
3. 将它的下游节点未满足依赖数减一。
4. 下游依赖数变成 0 时入队。
5. 所有节点处理完后得到顺序；若仍有节点无法处理，报告环等问题。

这是 Kahn 拓扑排序。此处的队列用于排列顺序，不代表已经实现了多线程并行调度。

计划还统计使用次数和 live values。当前执行器仍保留一个值名字到 Tensor 的字典，并返回该字典；它没有据此实现完整的逐值释放与全图内存地址规划。

## 5. prepare 和 run 各做什么

`OnnxExecutor::prepare(graph)` 编译并缓存执行计划，把 initializer 转为 Tensor 后保存。HushStreamSession 构造时对三张图分别 prepare。

这样同一流的每一帧不必重复转换全部权重。实际缓存使用图对象的地址作 key；图应保持稳定，且模型应比借用它的 Session 活得更久。

`run()` 创建这次调用的值字典，加入输入与权重，再按计划逐个访问节点。没有 prepare 的普通调用也能运行，但需要在调用时完成相应准备工作。

## 6. 分发：找到 op 对应的实现

在 `OnnxExecutor::run()` 的分支中查找 `node.op_type`：

| 类型 | 当前调用示例 |
| --- | --- |
| Conv、ConvTranspose | 解析属性后调用 Backend |
| ReLU、Sigmoid、Tanh | 调用 Backend |
| 相同形状的 Add/Mul | 调用 Backend |
| 广播 Add/Mul | 执行器中的 CPU 循环 |
| Reshape、Transpose、Concat | CPU 形状操作 |
| GRU | 执行器内的 ONNX GRU 实现 |
| 特定 Einsum、批量 MatMul | 部分用 Backend，部分直接 CPU 循环 |

所以选中 MetalBackend 后，模型中仍可能有 CPU 计算。判断执行位置，应沿实际分支查看。

算完后，将结果保存到 `tensors[node.outputs[0]]`。后续节点根据同一个名字查找它。

## 7. 流式状态为什么要单独传

`OnnxStreamState` 保存两张字典：`recurrent` 记录 GRU 隐藏状态，`padding` 记录因果 Pad 的历史输入。

流式执行中，一批数据结束时复制末尾状态到独立 Tensor。下一批开始时读取这些状态，而不是从零开始，也不能读到已被 pool reset 覆写的临时地址。

当前历史 Pad 支持特定 NCHW、左侧时间补零的模式。它不是对任意流式 ONNX 图都有效的自动转换器。

## 8. 支持算子名字不等于完整支持

例如当前 Cast 接受少量目标类型，但执行时仍沿用内部的 Float32 Tensor；Slice、Gather、Conv、GRU 也分别有维度、属性或布局限制。

`inspect_hush_graph()` 可以帮助筛出未认识的 op 名称，但它不能替代每个节点的形状和语义验收。迁移新模型时，必须同时检查 op、属性、dtype、形状和状态行为。

## 9. 自测

1. ONNX 读取器算不算模型输出？
2. 两个执行器是否都自动拓扑排序？
3. 把 CPUBackend 换成 MetalBackend，GRU 分支会自动改成 Metal shader 吗？
4. 为什么准备好的权重不应放进每帧 reset 的临时池？

答案：读取器只解析文件；仅 ONNX 路径有计划排序；不会，当前 ONNX GRU 有自己的 CPU 路径；权重需要跨调用稳定存在，不能让后续临时分配覆盖。
