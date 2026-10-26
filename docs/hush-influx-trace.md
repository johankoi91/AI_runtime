# Hush App 真实运行事件

打开 `build-app/bin/hush_mic_ui.app`，无需从终端设置 Influx 环境变量。默认连接当前演示服务器 `http://81.71.150.26/influx`。

1. 点击 `Generate Background Voice Demo`，等待生成结束；或 `Start Microphone` 后点击 `Record A/B Test`，自行开始与停止录音。
2. 顶部 Trace 行显示 collected、uploaded、dropped。`synced` 表示当前已采集事件上传完成；测试是否结束另看演示状态。
3. 点击 `Open Trace`，浏览器按本次 Run ID 打开真实事件区域。页面可手动刷新或每 10 秒刷新。

点击播放按钮只播放已有 WAV，不重新推理。麦克风测试仍需在系统设置中授权；应用更新签名后，macOS 可能要求重新授权。

## 记录范围

- `hush.app`：测试开始、结束、错误与来源（microphone / recording / background / real_probe）。
- `hush.frontend`：CPU Vorbis/STFT、ERB/DF 归一化及实际 shape。
- `hush.encoder`、`hush.erb_decoder`、`hush.df_decoder`：真实图执行阶段及节点详情。
- `hush.mask`、`hush.deep_filter`、`hush.synthesis`：CPU 增强和重建。
- `hush.state`：GRU/Pad 状态数、DF/合成历史字节数及临时池用量。
- `hush.batch`：一帧或一批的总耗时、实际帧数。批耗时不能解释成单帧延迟。

节点详情限前 1024 个节点。阶段记录前 3 批，之后单帧每 100 批、批处理每 4 批采样；底层已有 Metal 事件随阶段采样。前端与合成当前在 CPU 执行；节点按真实分支标记 CPU 或 Metal。时间是主机调用墙钟耗时，包含同步和采集开销，并非 GPU 硬件 profiler 结果。

上传仅包含计时、形状、计数和内存字节数，不包含录音样本或 Tensor 数值。采集端使用预分配的固定记录环形队列，emit 只复制原始字段和计时；不分配堆内存，不编码 Influx 文本，不做网络或文件 I/O。入队只尝试一次 try_lock，遇到锁竞争、队列满或字段过长时立即丢弃并计数；stage/event 各限 64 字节，detail 限 512 字节。这里优先保证 Hush 不等待，不保证事件零丢失。

后台 utility 优先级线程独立轮询队列（空队列约每 10 ms），完成转义、文本编码和不超过 64 KiB 的 HTTP 请求。预分配记录占用不超过默认 2 MiB，上报批文本额外最多 64 KiB；在途事件也占用队列额度，网络卡住不会无限积累。失败仅由后台重试。

销毁 InfluxDbSink 只设置结束标记，不 join、不 flush；后台通过共享状态持有自己的资源并尝试限时收尾。因此 App 切换测试、离线处理完成或关闭采集，都不等待网络。独立 CLI 验收工具可在推理结束后显式 flush；不要在音频回调、推理循环或持有音频锁时调用 flush。意外退出或长时间离线可能丢失未上传事件。

App 因现有服务器使用 HTTP/IP，包含 macOS ATS HTTP 访问例外；服务地址与认证可用 `INFLUXDB_URL/ORG/BUCKET/TOKEN` 覆盖。设置 `HUSH_TRACE_ENABLED=0` 可关闭 App 上传。

## 自测

```sh
cmake --build build-app --parallel
ctest --test-dir build-app --output-on-failure
node tests/web_trace_tests.js
python3 tests/verify_influx_trace.py --probe build-app/bin/influx_trace_probe
python3 tests/verify_influx_trace.py --probe build-app/bin/influx_trace_probe \
  --bundle "$PWD/build-app/models/hush-real-model.tar.gz"
HUSH_MODEL_BUNDLE="$PWD/build-app/models/hush-real-model.tar.gz" \
  HUSH_MEMORY_ITERATIONS=128 build-app/bin/hush_memory_tests
```

Runtime 回归检查阶段顺序、节点设备、异常事件、取消绑定及音频数值不变。HTTP 自测使用本地服务器检查并发序列、CSV 所需转义、失败重试、flush 与队列上限，不写入线上。网页自测覆盖多个 Influx 表、逗号/引号、微秒排序及 Run ID 隔离。

非阻塞回归会让本地 HTTP 服务收到请求后暂停响应，再执行 10 万次 emit、队列满压测和 sink 销毁；同一请求保持暂停时，再用真实 bundle 运行 32 帧 Metal Hush 推理。测试检查 emit 的堆分配为 0、p99 小于 1 ms、总循环小于 1 秒、析构小于 100 ms，并确认实际推理完成。一次本机结果为 emit 总计约 6.4 ms、p99 约 0.083 微秒、析构约 0.0002 ms；这些是该测试的采集端耗时，不是 Hush 推理延迟或硬实时保证。

2026-10-26 实测背景人声按钮产生 3158 条事件，3158 条上传成功、丢弃 0，网页显示 `REAL HUSH TRACE` 并正确定位 Run ID。真实模型 probe 产生 1598 条事件、全部上传；128 次单帧/128 次 32 帧内存测试约 23.7/28.7 MiB，未出现持续增长。用户重新授权后，麦克风已实际运行并上传事件。

实机录音 A/B 捕获 16.00 秒、256000 个 16 kHz 样本，7050 条事件全部上传、丢弃 0；Open Trace 定位到 `hush-ui-recording-7B99E813-C90F-4A13-AE49-82DEB8E7A6C1`，网页确认捕获、三张图、滤波、重建、状态与结束事件。初次长录音测试曾触发 Nginx HTTP 413，随后通过 64 KiB 分批修复并复验成功。

独立 ONNX 参考脚本的默认低幅度合成输入在 PCM 信噪比检查上得到 34.397 dB，未通过其 50 dB 门槛。隔离构建改动前的 b44161f 复现相同数值，改动前后导出 WAV 的 SHA-256 完全一致；此前的特征、子图及浮点音频误差检查均通过。此问题未在本次 trace 工作中修改，不声称该参考脚本全项通过。
