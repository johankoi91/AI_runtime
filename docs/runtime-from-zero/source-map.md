# 完整源码地图

这份地图按当前 Git 跟踪的 `include/`、`src/`、`apps/`、`tests/` 文件逐项列出。新增功能时应同步更新地图。核心阅读顺序见[课程入口](README.md)。

根目录的 [CMakeLists.txt](../../CMakeLists.txt) 决定哪些源文件会编译成库、工具、App 和测试；[README.md](../../README.md) 是项目入口。阅读一个文件时，还要确认它是否被当前执行入口调用。

最小 CPU 主线：`demo.cpp → Graph → Executor → CPUBackend → cpu::matmul/add → Tensor/MemoryPool`。

真实 Hush 主线：`hush_mic_ui.mm → HushStreamSession → HushFrontend / OnnxExecutor → CPU/Metal → DF/合成 → WAV/回放`。

## 数据与内存

| 文件 | 职责与阅读提示 |
| --- | --- |
| [include/runtime/dtype.h](../../include/runtime/dtype.h) | Float32 类型、元素大小和类型名 |
| [include/runtime/memory_pool.h](../../include/runtime/memory_pool.h) | arena 内存池接口与统计 |
| [include/runtime/storage.h](../../include/runtime/storage.h) | Buffer 所有权、地址、容量及对齐分配 |
| [include/runtime/tensor.h](../../include/runtime/tensor.h) | Tensor 的 dtype/shape/stride/offset/view 接口 |
| [src/memory_pool.cpp](../../src/memory_pool.cpp) | chunk 分配、对齐、reset 和已有块复用 |
| [src/tensor.cpp](../../src/tensor.cpp) | Tensor 分配、共享 view、连续布局和基础边界检查 |

## CPU 算子与后端接口

| 文件 | 职责与阅读提示 |
| --- | --- |
| [include/runtime/backend.h](../../include/runtime/backend.h) | CPU/Metal 共同执行接口 |
| [include/runtime/cpu_backend.h](../../include/runtime/cpu_backend.h) | CPUBackend 的接口声明 |
| [include/runtime/kernels.h](../../include/runtime/kernels.h) | CPU 算子函数声明 |
| [src/backend/cpu_backend.cpp](../../src/backend/cpu_backend.cpp) | 后端到 CPU kernel 的转发及部分序列实现 |
| [src/kernels/cpu/activations.cpp](../../src/kernels/cpu/activations.cpp) | Sigmoid、Tanh 等逐元素激活 |
| [src/kernels/cpu/add.cpp](../../src/kernels/cpu/add.cpp) | 相同形状连续 Tensor 的逐元素加法 |
| [src/kernels/cpu/conv2d.cpp](../../src/kernels/cpu/conv2d.cpp) | NCHW 卷积、步长、padding 和 groups |
| [src/kernels/cpu/conv_transpose2d.cpp](../../src/kernels/cpu/conv_transpose2d.cpp) | 分组反卷积、输出边界与 output_padding |
| [src/kernels/cpu/erb.cpp](../../src/kernels/cpu/erb.cpp) | 通用 ERB 频带计算和能量归一化 |
| [src/kernels/cpu/gru.cpp](../../src/kernels/cpu/gru.cpp) | 基础 GRU kernel；与 ONNX GRU 的语义存在差异，见第 3 课 |
| [src/kernels/cpu/istft.cpp](../../src/kernels/cpu/istft.cpp) | 通用 ISTFT 的 Tensor 包装 |
| [src/kernels/cpu/matmul.cpp](../../src/kernels/cpu/matmul.cpp) | 二维连续矩阵的乘加循环 |
| [src/kernels/cpu/mul.cpp](../../src/kernels/cpu/mul.cpp) | 相同形状连续 Tensor 的逐元素乘法 |
| [src/kernels/cpu/shape.cpp](../../src/kernels/cpu/shape.cpp) | reshape 共享 view、transpose 复制、concat |
| [src/kernels/cpu/spectrum.cpp](../../src/kernels/cpu/spectrum.cpp) | 将 ERB mask 投到复数频谱 |

## 手写图与 ONNX 执行

| 文件 | 职责与阅读提示 |
| --- | --- |
| [include/runtime/executor.h](../../include/runtime/executor.h) | 最小 Executor 接口和 logger/trace |
| [include/runtime/graph.h](../../include/runtime/graph.h) | 最小 Graph/Node/OpType，只含 Add/MatMul |
| [include/runtime/onnx_executor.h](../../include/runtime/onnx_executor.h) | prepare/run 与 OnnxStreamState 接口 |
| [include/runtime/onnx_plan.h](../../include/runtime/onnx_plan.h) | 拓扑计划、依赖和使用次数结构 |
| [include/runtime/onnx_weights.h](../../include/runtime/onnx_weights.h) | ONNX 图、节点、属性和读取接口 |
| [include/runtime/weights.h](../../include/runtime/weights.h) | 简易权重张量和自定义文件接口 |
| [src/executor.cpp](../../src/executor.cpp) | 按添加顺序执行最小 Graph |
| [src/graph.cpp](../../src/graph.cpp) | 按名字存取 Tensor 与添加节点 |
| [src/onnx_executor.cpp](../../src/onnx_executor.cpp) | 算子分发、CPU helper、ONNX GRU 和历史状态 |
| [src/onnx_plan.cpp](../../src/onnx_plan.cpp) | 构造依赖、拓扑排序、环检查与统计 |
| [src/onnx_weights.cpp](../../src/onnx_weights.cpp) | 手写 Protobuf 字段读取和部分 tensor 类型转换 |
| [src/weights.cpp](../../src/weights.cpp) | 简易权重文件读写和大小检查 |

## 音频与 Hush

| 文件 | 职责与阅读提示 |
| --- | --- |
| [include/runtime/audio.h](../../include/runtime/audio.h) | 音频格式、AudioFrame、AudioStream 和 WAV 接口 |
| [include/runtime/hush_bundle.h](../../include/runtime/hush_bundle.h) | 模型压缩包条目和读取接口 |
| [include/runtime/hush_frontend.h](../../include/runtime/hush_frontend.h) | 前端接口、历史采样和归一化状态 |
| [include/runtime/hush_model.h](../../include/runtime/hush_model.h) | HushConfig、三张图和模型契约 |
| [include/runtime/hush_session.h](../../include/runtime/hush_session.h) | 音频推理入口、状态 binding 和三张图的流状态 |
| [include/runtime/spectral.h](../../include/runtime/spectral.h) | 通用 Complex32/Spectrogram/STFT/ISTFT 接口 |
| [src/audio.cpp](../../src/audio.cpp) | 音频帧校验、流缓冲和固定帧提取 |
| [src/hush_bundle.cpp](../../src/hush_bundle.cpp) | 读取文件、zlib gzip 解压、tar 条目解析 |
| [src/hush_frontend.cpp](../../src/hush_frontend.cpp) | Hush Vorbis 窗、频谱、ERB/DF 特征与状态更新 |
| [src/hush_model.cpp](../../src/hush_model.cpp) | 配置解析、模型读取和算子名契约检查 |
| [src/hush_session.cpp](../../src/hush_session.cpp) | encoder/decoder 串联、DF、ERB mask、合成与 reset |
| [src/spectral.cpp](../../src/spectral.cpp) | 通用 Hann 窗与直接频率求和/重建 |
| [src/wav.cpp](../../src/wav.cpp) | 16 kHz 单声道等受限 PCM16 WAV 读写 |

## Metal、Trace 与观测

| 文件 | 职责与阅读提示 |
| --- | --- |
| [include/runtime/influxdb_sink.h](../../include/runtime/influxdb_sink.h) | 可选 InfluxDbSink 接口 |
| [include/runtime/metal_backend.h](../../include/runtime/metal_backend.h) | MetalBackend 的公开接口 |
| [include/runtime/trace.h](../../include/runtime/trace.h) | TraceSink 事件接口 |
| [src/backend/metal_backend.mm](../../src/backend/metal_backend.mm) | GPU shader 与 CPU 上的 Metal Buffer/提交/等待/回读 |
| [src/telemetry/influxdb_sink.mm](../../src/telemetry/influxdb_sink.mm) | 收集 trace、序列化并按配置上传 InfluxDB |

## 应用与命令行工具

| 文件 | 职责与阅读提示 |
| --- | --- |
| [apps/HushMicUI-Info.plist](../../apps/HushMicUI-Info.plist) | bundle 标识、版本、系统要求和麦克风用途 |
| [apps/demo.cpp](../../apps/demo.cpp) | 两节点 CPU 计算和手算提示；第 1 课入口 |
| [apps/hush_benchmark.cpp](../../apps/hush_benchmark.cpp) | 少量 32 时间步图执行计时与摊销指标 |
| [apps/hush_inspect.cpp](../../apps/hush_inspect.cpp) | 列出真实模型的节点和支持情况 |
| [apps/hush_interference_demo.mm](../../apps/hush_interference_demo.mm) | 两段中文合成语音的命令行演示 |
| [apps/hush_mic_demo.mm](../../apps/hush_mic_demo.mm) | 命令行麦克风采集与 Hush 示例 |
| [apps/hush_mic_ui.mm](../../apps/hush_mic_ui.mm) | 原生 App、采集/录制/处理/演示/回放 |
| [apps/hush_quality.cpp](../../apps/hush_quality.cpp) | WAV 长度、幅度等基础指标；非说话人质量评分 |
| [apps/hush_real_probe.cpp](../../apps/hush_real_probe.cpp) | 真实模型输入、图输出和增强入口诊断 |
| [apps/hush_speech_demo.h](../../apps/hush_speech_demo.h) | 读取独立语音、RMS 配平、SIR 混音和削波保护 |
| [apps/hush_stability.cpp](../../apps/hush_stability.cpp) | 重复 encoder 执行与有限值检查 |
| [apps/hush_wav_demo.cpp](../../apps/hush_wav_demo.cpp) | WAV → 流式 Hush → WAV |
| [apps/metal_demo.cpp](../../apps/metal_demo.cpp) | Metal 算子及图对照工具；需检查每项实际执行位置 |

## 测试与参考验收

| 文件 | 职责与阅读提示 |
| --- | --- |
| [tests/hush_memory_tests.mm](../../tests/hush_memory_tests.mm) | 真实模型 footprint、chunk 和输出有限值压力检查 |
| [tests/hush_reference_probe.cpp](../../tests/hush_reference_probe.cpp) | 导出特征、子图 Tensor 和最终音频供对照 |
| [tests/hush_reference_requirements.txt](../../tests/hush_reference_requirements.txt) | 可选 Python 参考验证的依赖列表 |
| [tests/runtime_tests.cpp](../../tests/runtime_tests.cpp) | 轻量测试框架及基础/ONNX/Hush 回归 |
| [tests/verify_hush_reference.py](../../tests/verify_hush_reference.py) | NumPy/ONNX Runtime 数值与整段音频参考 |

## 浏览器可视化

浏览器页面展示解释性动画和可选 trace 数据，不承担 Hush 数值推理。线上入口位于 `deploy/web/`：`index.html` 回放真实 Hush 事件，`tutorial.html` 提供数学与 GPU 动画课程；`inference.js`、`trace_data.js` 和 `tutorial*.js` 分别负责回放、数据解析和教程交互。

| 文件 | 职责与阅读提示 |
| --- | --- |
| [deploy/web/index.html](../../deploy/web/index.html) | 真实推理回放页面入口 |
| [deploy/web/inference.js](../../deploy/web/inference.js) | 推理阶段回放、事件查询与源码讲解 |
| [deploy/web/inference.css](../../deploy/web/inference.css) | 回放页面样式 |
| [deploy/web/trace_data.js](../../deploy/web/trace_data.js) | Influx CSV 解析、Run 隔离与阶段统计 |
| [deploy/web/tutorial.html](../../deploy/web/tutorial.html) | 18 节数学与 GPU 动画课程入口 |
| [deploy/web/tutorial-lessons.js](../../deploy/web/tutorial-lessons.js) | 课程内容、公式与源码映射 |
| [deploy/web/tutorial-math.js](../../deploy/web/tutorial-math.js) | 教学计算与动画使用的数学函数 |
| [deploy/web/tutorial.js](../../deploy/web/tutorial.js) | 教程交互、播放与绘图 |
| [deploy/web/tutorial.css](../../deploy/web/tutorial.css) | 教程页面样式 |

示意图中的节点状态或线程格子用于解释执行过程；判断当前实现时，应回到 `apps/hush_mic_ui.mm`、`src/hush_session.cpp` 和 `src/backend/metal_backend.mm` 的实际调用链。

## App 资源

| 文件 | 职责与阅读提示 |
| --- | --- |
| [apps/resources/README.md](../../apps/resources/README.md) | 测试语音的文本、合成来源和重新生成方法 |
| [apps/resources/hush-demo-background.wav](../../apps/resources/hush-demo-background.wav) | 内置中文合成干扰语音，非模型权重 |
| [apps/resources/hush-demo-target.wav](../../apps/resources/hush-demo-target.wav) | 内置中文合成主讲语音，非模型权重 |

## 如何使用这份地图

遇到音频问题，从 Hush/音频/App 三组向下查；遇到模型节点数值问题，从 ONNX 分发向 CPU/Metal 查；遇到内存问题，标出 Buffer 所有者、pool reset 和跨帧状态。浏览器可视化可以帮助观察流程，但判断当前实现应回到实际入口与调用链。
