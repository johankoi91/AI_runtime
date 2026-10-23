# 第 7 课：点一下按钮，发生了什么

[上一课](06-hush-pipeline.md) · [下一课](08-testing-and-boundaries.md)

源码入口：[hush_mic_ui.mm](../../apps/hush_mic_ui.mm)、[hush_speech_demo.h](../../apps/hush_speech_demo.h)、[CMakeLists.txt](../../CMakeLists.txt)。

## 1. App 是推理库的一个使用者

程序从 `main()` 创建 NSApplication 和 AppDelegate，进入事件循环。事件循环接收点击、窗口更新等事件，并调用注册的方法。

`applicationDidFinishLaunching()` 建窗口，读取环境变量或 App Resources 中的模型包，再通过 `loadModel()` 创建模型、Backend、Session 和 MemoryPool。

这几个对象的职责不同：Model 保存参数和图；Backend 执行计算；Session 保存某路音频的状态；pool 提供临时空间。

当前 Backend 由 `loadModel()` 根据下拉框创建。单纯改变下拉框选项，没有自动重建现有 Session 的逻辑；不能只看下拉框就断定已切换运行后端。这也是当前应用层仍需完善的一个边界。

## 2. 按钮到方法的对应关系

| 操作 | AppDelegate 方法 | 结果 |
| --- | --- | --- |
| Choose Model | `chooseModel:` → `loadModel:` | 加载 bundle 和创建执行对象 |
| Start/Stop Microphone | `toggle:` → `startMicrophoneEngine` | 请求权限、启动或停止采集 |
| Record A/B Test / Stop | `recordComparison:` | 开始记录，或停止后离线处理整段录音 |
| Generate Background Voice Demo | `generateInterferenceDemo:` | 读取内置两段语音，混音，再执行 Hush |
| Play Target Reference | `playTargetReference:` | 播放干净的演示主讲参考 |
| Play Mixed Input | `playMixedInput:` | 播放演示混音 |
| Play Enhanced Output | `playEnhancedOutput:` | 播放演示的 Hush 输出 |
| Play Raw Mic / Bypass / Hush | 对应 `play...:` → `playDemoFile:` | 播放麦克风三路 WAV |

播放按钮加载的是已有 WAV 文件。它们本身没有再运行一次模型。

## 3. 麦克风格式与模型格式

采集设备可能提供 48 kHz 等格式；模型要求 16 kHz 单声道。`AVAudioConverter` 负责格式转换，pending 数组负责将转换后的样本凑够 160 个一帧。

```text
设备回调的 PCM Buffer
    → AVAudioConverter
    → pending_audio_
    → 取出 160 samples
    → session_->process_audio
    → 更新输入/输出波形和统计
```

这个实时路径当前在音频 tap 中同步执行推理，没有独立的有界队列/后台推理线程设计。波形展示可用不等于已经具有成熟音频产品的调度与超时保障。

Start Microphone 主要展示采集、推理和波形，不会把系统输入设备替换成一个“降噪虚拟麦克风”。后者需要另外的设备/音频路由工作。

## 4. 为什么 A/B 录音另走一条路径

录制期间，tap 将设备格式的采样点存入 `native_recording_`；录制分支提前返回，不逐帧同步做 Hush。

用户停止录制后：

1. 移除 tap、停止采集，形成确定的录音边界。
2. 取得已采集的完整样本。
3. 在 serial recording queue 中转换为 16 kHz 单声道。
4. 使用独立的 offline Session，最多每 32 帧一批处理。
5. 输出裁到原始样本数，必要时做峰值保护并写 WAV。
6. 回主线程更新状态，再按当前逻辑恢复实时监看。

这避免了录音完整性依赖每一帧推理能否及时完成。较长录音仍会使用随时长增长的录音数组，它不是永远固定内存的磁盘流式录制器。

## 5. Raw、Bypass、Hush 的真实含义

| 文件 | 当前内容 |
| --- | --- |
| `/tmp/hush-mic-raw.wav` | 采集后转换为 16 kHz 单声道的音频 |
| `/tmp/hush-mic-bypass.wav` | 与 raw 相同，未做 Hush 增强 |
| `/tmp/hush-mic-enhanced.wav` | 相同长度的 Hush 输出，可能触发峰值保护 |

Raw 这个 UI 名称不表示保存了设备原始多声道/原始采样率文件。当前 Raw 和 Bypass 故意写同一组样本，用于检查重采样与播放链路。

## 6. 三种线程环境

| 环境 | 主要工作 | 当前同步方法 |
| --- | --- | --- |
| 主线程 | 窗口、按钮、定时刷新、播放控制 | 主线程事件循环 |
| 音频 tap | 接收采样、实时推理或记录采样 | 原子变量和 mutex 保护共享数据 |
| recording/demo 后台队列 | 完整录音处理、演示生成 | 完成后 dispatch 回主线程 |

`atomic` 适合简单标志与数值；`mutex` 适合多个字段或 vector 的一致读写；它们不会自动使任意模型、Backend 缓存都线程安全。

演示生成现在创建独立 Backend/Session，并持有模型副本，避免与实时路径共享同一组可写 Metal 临时缓存。

## 7. ARC 与 autoreleasepool

C++ 的 `unique_ptr/shared_ptr` 管 C++ 对象。Objective-C 对象主要由 ARC 管引用；autoreleased 临时对象需要在 autoreleasepool 排空时释放相应临时引用。

CMake 对 Objective-C++ 启用了 `-fobjc-arc`。音频 tap 和演示后台处理有 `@autoreleasepool`。这与 C++ pool 是两种机制，名字相似但职责不同。

定位内存问题时，应区分录音数组、CPU arena、Metal Buffer、Objective-C 临时对象，以及系统保留的虚拟地址空间。

## 8. 演示素材和模型参数

`make_hush_speech_demo()` 读取两段语音，按 RMS 归一化，用指定 SIR 混合，并在必要时对三路共同缩放防止输入削波。

Background 档位只改变生成的混音；改变档位后需重新 Generate。它不修改模型参数，也不会重新处理已录好的麦克风 WAV。

素材与来源见 [resources/README.md](../../apps/resources/README.md)。主讲参考用于判断被保留的声音，而不是作为模型的说话人注册输入。

## 9. 一个完整 App 包里有什么

```text
hush_mic_ui.app/Contents/
    MacOS/hush_mic_ui
    Info.plist
    Resources/hush-real-model.tar.gz
    Resources/hush-demo-target.wav
    Resources/hush-demo-background.wav
```

CMake 复制资源并做本地 ad-hoc 签名。它不等于 Apple 公证或正式开发者签名。资源复制完整和签名正确，都属于应用可启动的验证范围。

## 10. 自测

1. 为什么 Play Enhanced Output 与 Play Hush 可能播放两段不同的声音？
2. 为什么长录音的内存增长不能全部按“泄漏”判断？
3. 为什么在后台计算结束后，要回主线程更新按钮？
4. 把 demo 背景调到 0 dB 是否代表把模型降噪强度调到了 0？

答案：前者读演示文件，后者读麦克风录音文件；当前有随录音长度增长的样本数组，需与不应持续增长的计算内存区分；AppKit UI 操作应在主线程；不代表，0 dB 表示测试中两路语音 RMS 相同。
