# Hush 音频一致性验收

`Play Hush` 使用的增强链路必须与相同 ONNX 权重的参考计算一致。文件长度和有限值检查只能检查播放契约，无法发现时频排列、GRU 更新门或分块状态错误。

2026-10-24 修正了以下问题：

- `Pad` 按错误的维度顺序计算目标位置，打乱了卷积输入的时间与频率排列。
- ONNX GRU 更新门比例写反，并将 `sequence_lens` 的位置误当成 `initial_h`。同时补齐两种 `linear_before_reset` 语义。
- `ConvTranspose` 忽略 `output_padding`，少算频率边界；CPU 分组反卷积还混用了不同组的输入。
- 每 32 帧重新从零执行 GRU 和因果卷积。现在按节点保留隐藏状态与补零处的真实历史，并在 session reset 时清空。
- ERB 的对数稳定常数和归一化系数与参考前端不一致。

实时逐帧与离线批处理现在共用连续的流状态。最后不足 32 帧的批次按实际长度执行，避免多余补帧推进状态。状态拥有独立存储，内存池 reset 不会覆盖它们。

## 自测

先构建并运行不需要模型的回归测试：

```sh
cmake --build build-app -j4
ctest --test-dir build-app --output-on-failure
```

真实模型验收需要本地 bundle 和独立的 ONNX Runtime Python 环境，App 运行不依赖 Python：

```sh
python3 -m venv build-app/quality-venv
build-app/quality-venv/bin/python -m pip install -r tests/hush_reference_requirements.txt
build-app/quality-venv/bin/python tests/verify_hush_reference.py \
  --probe build-app/bin/hush_reference_probe \
  --bundle build-app/models/hush-real-model.tar.gz \
  --workdir build-app/quality/reference
```

默认使用固定随机种子的合成信号，包含不足 32 帧的最后一批。添加 `--input /absolute/path/input.wav` 可用自己的 16 kHz 单声道 PCM16 录音验证。录音和诊断文件都留在本机的输出目录。

检查项目：

1. NumPy 实现的 STFT、ERB、DF 归一化前端，最大误差小于 `2e-4`。
2. 三张 ONNX 子图输出的形状和数值，与 ONNX Runtime 对照，最大误差小于 `2e-4`。
3. CPU、Metal 各自使用 1 帧与 32 帧执行，均与完整序列参考结果对照。
4. 独立实现复数深度滤波和 Vorbis overlap-add 重建，最终浮点音频误差小于 `5e-5`。
5. 样本数一致，导出的 PCM16 音频与参考音频的误差信噪比超过 `50 dB`。

该信噪比衡量计算一致性，不是模型降噪量，也不等价于主观音质评分。参考脚本针对当前 16 kHz、320 点 FFT、32 ERB 频带、64 DF 频点、5 阶 DF、零 lookahead 的 bundle。

内存压力测试单独运行：

```sh
HUSH_MODEL_BUNDLE="$PWD/build-app/models/hush-real-model.tar.gz" \
  HUSH_MEMORY_ITERATIONS=512 build-app/bin/hush_memory_tests
```

## 本次验证记录

用户录音共 `91200` samples（`5.70s`）。四种后端/批大小组合均通过：

| 项目 | 结果 |
| --- | --- |
| 前端最大误差 | `8.74e-5` |
| 子图输出最大误差 | CPU `1.14e-5`，Metal `1.05e-5` |
| 最终浮点音频最大误差 | `3.09e-6` 以下 |
| PCM16 参考误差信噪比 | `58.98 dB` |
| 逐帧与 32 帧导出音频 | 完全一致 |
| 原始 / 修复前 / 修复后 RMS | `0.01673 / 0.00338 / 0.01466` |
| 内存压力测试 | 512 次逐帧及 128 次 32 帧批处理通过，约 `22–26 MiB` |

App 验收还需要检查 `Play Raw Mic`、`Play Bypass` 和 `Play Hush` 的播放时长相同。开始新的麦克风会话时重置流状态，避免沿用上一次采集的历史。
