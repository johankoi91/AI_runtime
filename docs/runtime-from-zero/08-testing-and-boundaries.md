# 第 8 课：什么证据才允许说“完成”

[上一课](07-macos-app.md) · [源码地图](source-map.md) · [返回课程入口](README.md)

源码入口：[runtime_tests.cpp](../../tests/runtime_tests.cpp)、[hush_memory_tests.mm](../../tests/hush_memory_tests.mm)、[verify_hush_reference.py](../../tests/verify_hush_reference.py)。

## 1. 把验收拆成五个问题

| 问题 | 例子 | 能证明什么 |
| --- | --- | --- |
| 数学语义正确吗 | 已知输入的 GRU 黄金数值 | 相应定义下的结果正确 |
| 接口和流程正确吗 | 输出 shape、样本数、状态连续性 | 当前调用契约成立 |
| 能及时处理吗 | 单帧耗时、分位数、回调积压 | 特定机器和负载下的性能 |
| 内存稳定吗 | 预热后 footprint 与 chunk 数 | 特定负载下没有持续异常增长 |
| 对用户有效吗 | 主讲保真、背景残留、主观听感 | 模型在目标场景的质量 |

通过第一项不自动通过第五项。文件可以完整播放，也可能内容已经失真。

## 2. 最小单元测试如何读

`runtime_tests.cpp` 用自己的 TestContext 和断言函数组织测试，不是 GoogleTest。先读某个 `test_...()`，再找它调用的实现。

推荐顺序：

1. `test_tensor_basics`、`test_add`、`test_matmul`：检查数据和基础计算。
2. `test_memory_pool_reuse`：检查 reset 后复用，包括多 chunk。
3. `test_graph_execution`：检查最小图闭环。
4. ONNX 解析与 execution plan 测试：检查读文件与依赖顺序。
5. `test_onnx_gru_reference`：与 ONNX Runtime 生成的已知数值对照。
6. `test_onnx_stream_padding`、`test_grouped_transpose_padding`：防止真实问题回归。
7. Hush Session、前端、ERB mask、ISTFT 测试：检查拼装后的契约。

测试数据的覆盖范围也有边界；一个例子通过不代表所有属性、形状和坏输入都通过。

## 3. CPU 与 Metal 相互对照还不够

如果两者共用了同一段 CPU 代码，或者复制了同一个错误公式，数值完全一致仍可能都错。

基础 GRU 曾经就是值得警惕的路径：Metal 方法可以直接返回 CPU 方法；演示的“Metal GRU passed”不能单独证明已执行 GPU shader 或符合 ONNX 语义。

独立参考可以是手算、另一套算法、标准定义的固定值、官方前端库或外部推理框架。Hush 参考脚本采用 NumPy 前端/重建与 ONNX Runtime，诊断时也对照过官方 DeepFilterLib。

## 4. 真实模型参考检查

`hush_reference_probe.cpp` 导出特征、各图中的 Tensor 和最终音频。`verify_hush_reference.py` 用同一个 bundle 在 ONNX Runtime 中计算，再比较形状、数值与音频。

它覆盖 CPU/Metal、逐帧/32 帧两种调用方式，以及最后不足整批的情况。它依赖真实模型和 Python 包，和最小 C++ 测试不同。

当前脚本的固定误差阈值与特定 16 kHz 配置有关。其他录音可能触发更大的独立前端误差，不能只修改阈值让它“变绿”；应先区分频谱误差、归一化累计误差与图执行误差。固定合成输入或历史短录音通过，不等于所有真实音频都已覆盖。

方法及历史结果见 [Hush 音频验收](../hush-audio-validation.md)。

## 5. 测内存看哪些数

`hush_memory_tests.mm` 同时观察 pool 的 chunk 数与进程 physical footprint，在预热后持续执行，并多次 reset Session。

```sh
HUSH_MODEL_BUNDLE="$PWD/build-app/models/hush-real-model.tar.gz" \
  HUSH_MEMORY_ITERATIONS=512 ./build-app/bin/hush_memory_tests
```

模型路径需要实际存在。没有环境变量时，该测试返回跳过码 77；CTest 显示 Skipped 不表示模型内存已经测过。

虚拟地址空间、RSS、physical footprint、arena 使用量有不同含义。不要看到 VSZ 数百 GB 就直接认定占用了同样大的物理内存，也不要只看 pool 的 used bytes 就断定整个 App 稳定。

## 6. 如何解读 benchmark

当前 `hush_benchmark` 以 32 个时间步运行 encoder 和 decoder，迭代次数较少。输出中的 `per_10ms_ms` 是整批时间除以 32 得到的摊销值。

```text
一批含 320 ms 音频，模型耗时 64 ms
    → 摊销 2 ms / 每 10 ms 音频
```

这个数不能直接证明：每次单帧请求都在 2 ms 内返回、总端到端延迟只有 2 ms、长期 p99 满足 10 ms，或者麦克风回调永不积压。benchmark 也没有完整覆盖前后端和设备缓冲。

优化前先固定输入、批大小、机器、编译配置和度量范围。历史 `301 ms/frame` 与后来的批摊销值，不能在没有说明口径时直接比较。

## 7. 从症状回到正确层级

| 症状 | 优先检查 |
| --- | --- |
| 播放很快、时长变短 | 采样率、转换输入回调、样本数、WAV 头 |
| 长时间推理内存持续增长 | arena 复用、缓存增长、ARC、autoreleasepool |
| CPU/Metal 图输出不同 | 找到第一个不一致节点，检查属性、布局和 kernel |
| 分块边界出现变化 | GRU、Pad、DF、归一化和合成历史 |
| 与参考一致但主讲变糊 | 模型误抑制、输入分布、主讲保真与音质验收 |

“第一个不一致节点”比只比较最终 RMS 更容易定位原因。RMS 变小只说明整体能量下降，无法单独证明背景人声被选择性去掉。

## 8. 当前仍值得继续完成的工程能力

- 更完整的 Tensor 布局与访问边界检查。
- 统一并验证基础 GRU API 与 ONNX GRU 的语义。
- 扩展 ONNX 的属性、dtype、动态形状与错误路径覆盖。
- 减少 CPU/GPU 复制、逐算子等待与命令提交。
- 根据使用寿命规划中间张量空间，控制缓存容量。
- 有界音频队列、线程安全、多流/后端切换和长期负载测试。
- 为目标说话人保留建立真实音频质量门槛。

这些是当前源码可继续演进的方向，不是本课要求你立即实现的作业。

## 9. 你怎样证明自己读懂了

先任选一条已有路径，做下面五项练习：

1. 画出输入到输出的调用图，标出文件和函数。
2. 预测主要 Tensor 的 shape、元素数和字节数。
3. 标出每块数据的所有者，以及 reset 后的有效期。
4. 指出真正的 CPU/GPU 执行位置和一次同步点。
5. 设计一个能发现真实错误的输入，而不是只检查程序不崩溃。

一开始用两节点 demo 完成这五项；最后用 `process_audio_batch()` 再完成一次。两次都能解释清楚，就建立起了从基础 runtime 到真实音频应用的完整理解。
