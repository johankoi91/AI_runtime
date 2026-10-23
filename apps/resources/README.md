# Hush 双人语音测试素材

这两段是使用 macOS 内置语音合成生成的中文测试语音，均为 16 kHz、单声道 PCM16。它们用于可重复的功能演示，不代表真人录音、训练数据或覆盖各类说话人的质量基准。

- `hush-demo-target.wav`：Tingting，语速 185。
- `hush-demo-background.wav`：Meijia，语速 195。

主讲内容：

> 这是一段主讲人的测试语音。请保留我的声音，让每一个字都清晰自然。我们正在比较背景人声处理之前和处理之后的效果。

背景内容：

> 今天下午我们准备出门买东西，先去超市，再看看附近有没有咖啡店。如果天气不错，晚上可以和朋友一起散步聊天。

App 可选择主讲/背景 RMS 比为 0、3、12、18、24 dB。18 dB 是默认测试条件，0 dB 是同音量挑战条件。这个选择只改变生成的混音，不改变 Hush 模型，也不影响麦克风录音。

生成示例：

```sh
say -v Tingting -r 185 -o target.aiff '这是一段主讲人的测试语音。请保留我的声音，让每一个字都清晰自然。我们正在比较背景人声处理之前和处理之后的效果。'
afconvert -f WAVE -d LEI16@16000 -c 1 target.aiff hush-demo-target.wav
say -v Meijia -r 195 -o background.aiff '今天下午我们准备出门买东西，先去超市，再看看附近有没有咖啡店。如果天气不错，晚上可以和朋友一起散步聊天。'
afconvert -f WAVE -d LEI16@16000 -c 1 background.aiff hush-demo-background.wav
```
