# 第 3 课：找到真正算数的代码

[上一课](02-tensor-and-memory.md) · [下一课](04-graphs-and-onnx.md)

源码入口：[kernels.h](../../include/runtime/kernels.h)、[CPUBackend](../../src/backend/cpu_backend.cpp)、[CPU kernels 目录](../../src/kernels/cpu)。

Kernel 可以先理解为“执行一种基础计算的函数”。算子的数学定义、支持的输入形状、数字怎样摆放，是三个不同的问题。

## 1. Add 和 Mul：同位置的数字一起算

`cpu::add()` 对相同形状的两个连续 Float32 Tensor 做：

```text
output[index] = left[index] + right[index]
```

`cpu::mul()` 则把加号换成乘号。它们先检查约束，再分配输出，再访问数据。

这两个底层函数要求形状相同。ONNX 中 `[2,3] + [3]` 的广播处理在执行器的 `broadcast_add()` 中，不是底层 Add 自动完成的。

“广播”表示按规则复用较小输入的值，并不一定真地把它复制成大数组。

## 2. MatMul：三个循环各管什么

在 [matmul.cpp](../../src/kernels/cpu/matmul.cpp) 中，对 `[M,K] × [K,N]` 的输入，输出形状为 `[M,N]`。

源码的循环顺序是输出行 → 相乘相加的中间维 → 输出列：

```text
for each row
    for each inner
        for each column
            output[row,column] += left[row,inner] × right[inner,column]
```

`inner` 是把一行和一列加起来所走的方向。例如 `[2,3] × [3,2]`，每个输出值累加 3 项。

内存是一维的，所以真实访问使用：

```text
left[row × K + inner]
right[inner × N + column]
output[row × N + column]
```

该实现用普通 C++ 乘加循环，不是分块 BLAS 内核。正确实现基础矩阵乘法，与达到成熟框架的 MatMul 性能，是不同的验收目标。

## 3. 激活：改变每个数的范围

| 函数 | 对数字的作用 | 所在位置 |
| --- | --- | --- |
| ReLU | 负数变 0，非负数保留 | CPUBackend 的 `relu()` |
| Sigmoid | 映射到 0～1，常作为门控比例 | `activations.cpp` |
| Tanh | 映射到 −1～1 | `activations.cpp` |

它们通常不改变 Tensor 的形状。形状不变不代表内容不变。

## 4. 形状操作也会影响计算

`shape.cpp` 中的 reshape、transpose、concat，与 ONNX 执行器内的 Pad、Slice、Gather 共同负责整理数据。

- reshape：按新形状解释同一组连续数字。
- transpose：交换维度顺序，当前实现会搬运数字。
- concat：沿指定维把多个输入接起来。
- Pad：在边缘补数值；流式因果卷积还要使用真实历史。
- Slice：选取一段范围。
- Gather：按给定的索引取数据。

这些操作即使没有神经网络权重，也会决定后面的卷积读到了什么。之前 Pad 的目标下标算错，就曾把时间和频率排列打乱。

## 5. Conv：在局部窗口做乘加

阅读 [conv2d.cpp](../../src/kernels/cpu/conv2d.cpp)。当前主要使用 NCHW：

```text
N：一批有几个样本
C：每个样本有几个通道
H：高度，在 Hush 特征里常表示时间
W：宽度，在 Hush 特征里常表示频率
```

卷积在输入局部区域取一组数字，与一组权重对位相乘再相加。stride 决定窗口每次移动多远；padding 决定边缘补多少数。

例如宽度为 32、卷积核宽度为 3、左右各补 1、步长为 2：

```text
输出宽度 = floor((32 + 2×1 − 3) / 2) + 1 = 16
```

`groups` 决定通道分组。groups 等于输入通道数且权重通道配置匹配时，可进行逐通道卷积；每组只读本组的输入，不能混入其他组。

## 6. ConvTranspose：生成更大的输出网格

阅读 [conv_transpose2d.cpp](../../src/kernels/cpu/conv_transpose2d.cpp)。它不是“求卷积的数学逆”，也不保证恢复原始数据。

当前无膨胀的输出宽度公式为：

```text
(输入宽度 − 1) × stride + kernel_width − 2×padding + output_padding
```

从 8 扩到 16 的例子：`(8−1)×2 + 3 − 2×1 + 1 = 16`。少处理最后的 output_padding，就会少一个频率位置，后续相加也会出错。

## 7. GRU：当前输入加上一份历史记忆

GRU 处理序列。每一帧既看当前输入，也看前一帧的隐藏状态。隐藏状态是一组数字，不是磁盘上的日志。

ONNX GRU 的更新式是：

```text
当前状态 = update × 旧状态 + (1−update) × 候选状态
```

update 接近 1 时保留更多旧状态。把两个系数写反，即使形状与输出长度完全正确，结果仍然会变。

当前有两条需要分开阅读的实现：

| 路径 | 入口和状态 |
| --- | --- |
| 基础 Backend GRU API | `cpu::gru_step()`、`CPUBackend::gru_sequence()` 和 MetalBackend 对应方法；保留简化/实验实现 |
| 真实 Hush ONNX GRU | [onnx_executor.cpp](../../src/onnx_executor.cpp) 内 `run_gru_step()`；处理 ONNX 初始状态、两种 `linear_before_reset`，并有参考数值测试 |

不能把基础 GRU API 当成完整 ONNX GRU 的替代。源码中基础 `gru_step()` 仍有 reset 未用于候选状态、更新公式与 ONNX 不同的问题；“CPU/Metal 相互一致”没有证明它符合通用 GRU 定义。Hush 当前走的是第二条已单独验证的路径。

`linear_before_reset` 决定 reset 门是在循环矩阵乘法之前，还是之后参与候选状态计算。两个位置通常不能交换，因为矩阵乘法会混合不同单元。

## 8. 音频算子

`erb.cpp` 将频谱按频带聚合、提供归一化函数；`spectrum.cpp` 将 ERB mask 投回频点并乘到复数频谱上；`istft.cpp` 连接通用的逆短时频谱重建。

真实 Hush 路径另外使用 `hush_frontend.cpp` 的窗口与归一化，以及 `hush_session.cpp` 的深度滤波和合成。第 6 课解释为什么不能只看到后端存在某个函数，就认为 Hush 一定调用它。

## 9. 自测

1. 两个 `[2,3]` 数组逐元素乘法的输出形状是什么？
2. `[2,3] × [2,4]` 是否满足当前二维 MatMul 的条件？
3. 为什么漏掉一个频率位置也会影响音质？
4. 两个实现给出相同结果，是否足以证明数学公式正确？

答案：`[2,3]`；不满足，内维 3 和 2 不相等；后续算子的频率位置、形状和边界值都会改变；不足，还需要独立定义、手算值或外部参考实现。
