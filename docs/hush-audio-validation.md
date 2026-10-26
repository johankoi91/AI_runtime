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

App 内的 `Generate Background Voice Demo` 现在使用两段独立中文语音。默认混音为主讲人高于背景人声 `18 dB`，还可以选择 `24/12/3/0 dB`；生成后依次播放 `Play Mixed Input` 和 `Play Enhanced Output`。主讲参考仍写入 `/tmp`，用于诊断工具。

模型的公开训练范围是背景人声低于主讲人约 `12–24 dB`。同音量或背景更响属于挑战场景，当前 8 MB 模型不保证完全分离；其分离 head 只用于训练正则，未导出为独立的背景源分离输出。Hush 的结果与 ONNX Runtime 已对齐，因此主观效果受模型能力和输入声学条件限制。

训练范围也不构成效果保证。本机用两段独立中文合成语音做的受控测试，在 `12/18/24 dB` 输入条件下没有得到稳定的主讲分离收益，单人对照也有明显语音损伤。这是该组测试素材的结果，不能外推为所有中文或真人录音的结论。参考 `apps/resources/README.md` 中的文本与声音来源可重复生成素材。新版演示改进了验证方式，没有增加新的说话人分离模型。

仅观察增强后 RMS 降低不能证明背景人声得到选择性抑制。需要同时检查主讲保留、干扰残留和重建失真，并补充真人语音、独立源及具体设备/距离条件后，才能评估面向目标场景的质量。
