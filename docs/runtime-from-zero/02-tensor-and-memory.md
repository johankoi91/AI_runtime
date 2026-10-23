# 第 2 课：数字放在哪里，谁负责释放

[上一课](01-first-computation.md) · [下一课](03-cpu-kernels.md)

源码入口：[storage.h](../../include/runtime/storage.h)、[tensor.h](../../include/runtime/tensor.h)、[tensor.cpp](../../src/tensor.cpp)、[memory_pool.cpp](../../src/memory_pool.cpp)。

## 1. 先理解地址和字节

程序运行时，数字存在内存中。地址用来找到那块内存；字节是计算空间大小的单位。本项目的 Float32 是一个 4 字节浮点数。

假设六个数字连续存放：

```text
元素下标      0   1   2   3   4   5
数字          1   2   3   4   5   6
相对字节偏移  0   4   8  12  16  20
```

内存本身没有“这是 2 行 3 列”的标签。这个解释来自 Tensor 的元信息。

## 2. Buffer 管实际空间

`Buffer` 的三个字段分别是：

| 字段 | 作用 |
| --- | --- |
| `owner` | 共享的所有权，决定底层空间何时释放 |
| `data` | 第一个字节的地址 |
| `bytes` | 这一段空间有多少字节 |

`make_owned_buffer()` 调用带对齐要求的 `operator new` 分配空间，并把释放动作交给 `shared_ptr<void>` 的 deleter。最后一个所有者销毁时，deleter 释放这块空间。

`shared_ptr` 可以先理解成“多人共同持有一张空间凭证”。复制凭证通常不会复制整块数字，只有最后一张凭证消失时才释放空间。

## 3. Tensor 管解释方法

Tensor 在 Buffer 外加上这些信息：

```text
dtype           Float32，每个元素如何解释
shape           [2,3]，两行三列
strides         [3,1]，沿各维走一步跨过几个元素
offset_elements 0，从底层空间的第几个元素开始
```

对于标准连续二维数组：

```text
元素位置 = offset + row × stride[0] + column × stride[1]
字节位置 = 元素位置 × 4
```

例如 `[1,2]` 的元素位置为 `0 + 1×3 + 2×1 = 5`，数字是 6，距离底层起点 20 字节。

`Tensor::data<float>()` 已经把 Tensor 自身的 offset 算进返回地址。使用这个返回地址再索引时，不要把相同 offset 重复加一次。

## 4. 三种“创建”行为

| 调用 | 空间与数据发生什么 |
| --- | --- |
| `Tensor::empty(shape)` | 分配独立空间，不初始化数字 |
| `Tensor::zeros(shape)` | 分配独立空间，再填 0 |
| `Tensor::from_vector(shape, values)` | 分配独立空间，再复制 values |
| `Tensor::view(...)` | 共享已有 Buffer，修改解释方式 |
| `Tensor copied = original` | 复制元信息和共享所有权，通常不深拷贝数字 |

因此，给 `copied.data<float>()[0]` 写入新值，可能会改变 `original` 读到的内容。需要独立数字时，要明确分配空间并复制。

## 5. reshape、view、transpose 的当前实现

阅读 [shape.cpp](../../src/kernels/cpu/shape.cpp)。

`cpu::reshape()` 要求输入连续、元素总数相同，然后调用 `view()`。把 `[2,3]` 改成 `[3,2]`，仍然是按原来的线性顺序解释六个数字。

`cpu::transpose()` 当前会分配输出并逐元素搬运，得到新的连续数组。这和一些成熟框架通过修改 stride 创建转置 view 的实现不同。

```text
原始 [2,3]    reshape [3,2]   transpose [3,2]
[1 2 3]       [1 2]           [1 4]
[4 5 6]       [3 4]           [2 5]
              [5 6]           [3 6]
```

所以“形状一样”并不说明数字顺序一样；“API 叫 transpose”也不自动说明它零拷贝。

当前 `Tensor::validate()` 只做基础检查，未完整验证任意 stride 的访问边界；`fill()` 也是线性写入实现。学习阶段应按各算子的连续输入约束使用，不能把能创建某种 view 当成所有函数都支持它。

## 6. 内存池为什么有用

一帧音频会产生很多临时张量。如果每帧都申请、释放相似大小的空间，分配开销和碎片可能变大。

当前 `MemoryPool` 使用 arena 方式：先申请较大的 chunk，再从 chunk 内顺序分配小段空间。每个小段的 Buffer 共享大块空间的所有权。

```text
chunk：预留一块较大的空间
┌───────────────┬──────────────┬────────────────┐
│ hidden，16 B  │ out，16 B    │ 尚未使用        │
└───────────────┴──────────────┴────────────────┘
                               ↑ offset
```

`allocate_tensor()` 先根据形状算字节数，再调用 `allocate()`。`align_up()` 将位置推进到符合对齐要求的边界。统计中的 used 包含实际推进的部分，不等同于预留总容量。

## 7. reset 的关键含义

`reset()` 把各 chunk 的 offset 和当前 used 归零，保留 chunk，以便下轮复用。

```text
reset 前：空间里有上一轮的数字
reset 后：数字可能还在，但这段位置已允许重新分配
下一轮：同一地址可能被写成别的结果
```

因此，底层空间尚未被释放，不等于上一轮 Tensor 的数值仍然有效。跨帧 GRU 状态必须保存到独立拥有的空间，不能只留下一个指向临时池的 view。

Hush 中 `OnnxStreamState` 的隐藏状态和历史补零数据采用独立 Tensor，就是为了解决这个问题。

## 8. 曾经内存暴涨的机制

旧的 `reserve_chunk()` 只查看最后一个 chunk。假设一次推理需要三个 chunk，reset 后再执行时，它只从第三块开始；第三块满了又追加新块，前两块没有获得复用。

当前实现会扫描已有 chunk。于是同类请求在 reset 后可以复用前面的空间。`test_memory_pool_reuse` 包含多 chunk 回归检查。

这个例子说明：有 `reset()` 这个函数名，不代表整套分配逻辑已经正确复用。

## 9. 三类生命周期

| 数据 | 应该保留多久 | 当前存放方式 |
| --- | --- | --- |
| 模型权重 | 模型/执行器生命周期 | 模型参数和 prepare 后的 Tensor 缓存 |
| 本轮中间结果 | 一帧或一批处理完成前 | MemoryPool |
| GRU、归一化、卷积历史 | 同一音频流跨帧保存 | Session 状态和独立 Tensor/vector |

`unique_ptr` 表示单个所有者；`shared_ptr` 表示共享所有者；引用和原始指针通常表示借用。借用者不会自动延长被借用对象的寿命。

## 10. 自测

1. `[2,3,4]` 的 Float32 连续 Tensor 有几个元素、多少字节、默认 stride 是多少？
2. 复制一个 Tensor 对象，是否一定复制整块数字？
3. reset 后 retained Tensor 的 shared_ptr 仍存在，为什么还可能读到新一帧数据？
4. 本项目 transpose 是否只改变元信息？

答案：24 个、96 字节、`[12,4,1]`；不一定，默认复制共享 Buffer；因为 pool 已允许重新使用同一段位置；当前 transpose 会分配并复制数字。
