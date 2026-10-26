export const phaseOrder = ["input", "frontend", "encoder", "erb", "df", "mask", "deep_filter", "synthesis", "state", "output"];
export const measuredPhases = ["frontend", "encoder", "erb", "df", "mask", "deep_filter", "synthesis"];
export const phases = {
  input: {
    title: "音频输入与分帧", device: "CPU", source: "apps/hush_mic_ui.mm · HushStreamSession::validate_frame",
    explanation: "麦克风原始采样率先转换为 16 kHz 单声道，再切为每帧 160 个采样点。离线录音可以把最多 32 帧送入一次模型调用。",
    input: "Mic / WAV · native samples", output: "AudioFrame · [160] / frame",
    note: "160 / 16000 = 10 ms 是这帧音频的时长；处理它花了多久，要看独立记录的推理耗时。"
  },
  frontend: {
    title: "前端：波形变成两路特征", device: "CPU", source: "src/hush_frontend.cpp · HushFrontend::process",
    explanation: "上一帧和当前帧组成 320 点分析窗，经 Vorbis 窗与频谱计算得到 161 个复数频点，再构造 ERB 能量和低频 DF 特征，更新归一化历史。",
    input: "AudioFrame [160] + analysis history [160]", output: "spectrum [1,161,2] · ERB [1,32] · DF [2,1,64]",
    note: "当前只为完整 HushFrontend 调用计时。STFT、ERB、DF 内部没有独立计时点；实部与虚部也不是两个人或两个声道。"
  },
  encoder: {
    title: "Encoder：编码声音特征", device: "CPU + Metal", source: "src/hush_session.cpp · process_encoder → OnnxExecutor::run",
    explanation: "enc.onnx 读取 ERB / DF 两路 Tensor，按执行计划依次运行节点。支持的 Conv、激活等通过 MetalBackend 执行；ONNX GRU 和部分形状、广播运算由 CPU 执行。",
    input: "feat_erb [1,1,T,32] · feat_spec [1,2,T,64]", output: "e0 / e1 / e2 / e3 · emb · c0 · lsnr",
    note: "Encoder 的输出同时供两张 decoder 使用。节点详情中的 device 字段来自实际分发分支；不能因为选择了 Metal backend，就把所有节点当成 GPU。"
  },
  erb: {
    title: "ERB decoder：预测频带增益", device: "CPU + Metal", source: "src/hush_session.cpp · process_decoders → erb_dec.onnx",
    explanation: "ERB decoder 读取 Encoder 的中间特征和表征，生成每帧 32 个频带的增益 m，供后续频谱增强使用。",
    input: "Encoder named outputs", output: "m · [1,1,T,32]",
    note: "m 是网络预测的结果，不是演示里的手动混音参数。该 decoder 与 DF decoder 共享 Encoder 输出，当前执行顺序是先 ERB 再 DF。"
  },
  df: {
    title: "DF decoder：预测复数滤波系数", device: "CPU + Metal", source: "src/hush_session.cpp · process_decoders → df_dec.onnx",
    explanation: "DF decoder 生成低频 64 个频点的 5 阶复数滤波系数，后面用它们给历史频谱和当前频谱做加权组合。",
    input: "Encoder named outputs", output: "coefs · [1,T,64,10]（5 阶 × 实/虚）",
    note: "这条支路接收 Encoder 输出，不读取 ERB mask 作为输入。两路增强结果随后在频率范围上组合。"
  },
  mask: {
    title: "ERB mask：把增益映射回频点", device: "CPU", source: "src/hush_session.cpp · apply_mask",
    explanation: "将 32 个 ERB 频带的增益投回 161 个频点，并与当前频谱相乘。最终高频部分采用这一路结果。",
    input: "原始 spectrum [1,161,2] + m [1,32]", output: "masked spectrum [1,161,2]",
    note: "这一步是 CPU 频谱处理。它消耗模型输出的 mask，没有在本页面上调整模型音质的滑块。"
  },
  deep_filter: {
    title: "Deep filter：利用低频历史", device: "CPU", source: "src/hush_session.cpp · apply_deep_filter",
    explanation: "对低频 64 个频点，将 4 帧历史和当前原始复数频谱按 5 阶系数求和；再把高频替换为 ERB mask 的结果。",
    input: "原始 spectrum + coefs + DF history", output: "enhanced spectrum [1,161,2]",
    note: "低频深度滤波读取原始频谱，不是先把所有频点 mask 一遍再做 DF。这里记录真实计算路径，避免把简化示意当成完整算法。"
  },
  synthesis: {
    title: "ISTFT / overlap-add：恢复连续音频", device: "CPU", source: "src/hush_session.cpp · synthesize_hush",
    explanation: "增强频谱经过逆变换、Vorbis 合成窗和历史重叠相加，每帧得到 160 个连续采样点，留存下一帧需要的合成历史。",
    input: "enhanced spectrum [1,161,2] + synthesis history", output: "AudioFrame [160] / frame",
    note: "当前完整 Hush session 使用 CPU 参考重建。Metal ISTFT 原语存在，但此路径没有调用它。音频必须保持原始采样率和帧序。"
  },
  state: {
    title: "历史保留：下一帧继续推理", device: "CPU", source: "include/runtime/onnx_executor.h · OnnxStreamState / HushStreamSession",
    explanation: "GRU 隐状态、causal Pad 历史分别由三张图的 OnnxStreamState 持有。前端归一化、DF 和合成历史由各自模块保存。",
    input: "当前帧 / 批次的历史更新", output: "下一帧的独立持久状态",
    note: "pool.reset() 只能复用临时内存。持久状态必须独立存储，否则下一帧写入临时池时会覆盖历史。state_retained 是汇总观测点，不代表所有状态都在这一刻才更新。"
  },
  output: {
    title: "输出：本帧 / 批次推理返回", device: "CPU", source: "src/hush_session.cpp · process_audio_batch",
    explanation: "本次调用返回一帧或一批 AudioFrame，App 可以继续写入 WAV、更新波形或播放。下一次调用沿用刚才保留的流状态。",
    input: "合成完成的音频帧", output: "T × 160 samples · 16 kHz mono",
    note: "整批耗时包含本次调用的前端、网络、频谱增强和重建。32 帧批次对应 320 ms 音频；除以 32 只能得到摊销成本，不能当作单帧响应时间。"
  }
};

export const calculations = {
  input: {
    title: "先把连续声音切成小块",
    formula: "16000 samples / second\n160 samples / frame = 0.010 second = 10 ms",
    steps: ["读取原始采样，转成 16 kHz、单声道。", "每次取 160 个 float，组成一个 AudioFrame。", "帧尾不足时补零或等待下一批；模型只接受固定长度。"],
    note: "160 是时间轴上的采样数量，不是 160 个字，也不是 GPU 线程数。"
  },
  frontend: {
    title: "STFT：每个频点有多少能量",
    formula: "xw[n] = x[n] × VorbisWindow[n]\nX[k] = Σ(n=0..319) xw[n] × e^(-i·2πkn/320)\n频点 k = 0..160，所以得到 161 个复数",
    steps: ["上一帧 160 点 + 当前帧 160 点，组成 320 点分析窗。", "每个频点都把 320 个样本乘上不同的正弦 / 余弦并求和。", "实部和虚部一起表示该频率的强度与相位。", "再按 ERB 频带聚合能量，并提取前 64 个 DF 复数频点。"],
    note: "这里的实部 / 虚部不是左右声道；它们是同一个频率的两个数学分量。"
  },
  encoder: {
    title: "Encoder：重复的张量乘加",
    formula: "Conv / Linear 的核心：\ny[out] = bias[out] + Σ(input[i] × weight[out,i])\nGRU 的核心：h[t] = GRU(x[t], h[t-1])",
    steps: ["每个 ONNX 节点读取上游 Tensor，按依赖顺序运行。", "卷积 / 线性层对输入做很多次乘法和加法。", "激活函数改变结果的非线性；GRU 还读取上一帧的 hidden state。", "节点输出写回执行器的 Tensor 字典，命名输出交给两张 decoder。"],
    note: "同一个 Encoder 里既可能有 Metal 节点，也可能有 CPU fallback；看节点事件中的 device 才能知道实际位置。"
  },
  erb: {
    title: "ERB decoder：从特征预测 32 个增益",
    formula: "m = erb_decoder(e0, e1, e2, emb, c0, lsnr)\n输出：m[band]，每个 band 对应一个听觉频带",
    steps: ["读取 Encoder 的命名输出。", "网络内部仍是 Conv / Linear / 激活等张量运算。", "最终得到当前帧或批次的 32 个频带增益。", "这个 m 之后会投影回 161 个频点。"],
    note: "页面显示的是网络输出契约；不要把 m 当成一个可手动调节的 UI 参数。"
  },
  df: {
    title: "DF decoder：预测复数系数",
    formula: "coefs[k,j] = df_decoder(encoder_outputs)\nj = 0..4：5 个历史 tap\n每个 tap = real + i·imag",
    steps: ["读取同一份 Encoder 输出。", "网络为低频 64 个频点预测 5 阶复数系数。", "每个复数系数后面会参与一次复数乘法。", "系数本身不是音频；它描述怎样组合历史频谱。"],
    note: "5 阶意味着当前频谱加上最多 4 帧历史；第一帧历史不足时使用零。"
  },
  mask: {
    title: "把 32 个频带增益投影回 161 个频点",
    formula: "gain[k] = Σ(b=0..31) projection[k,b] × m[b]\nS_mask[k] = S[k] × gain[k]",
    steps: ["每个频点 k 查找它与 32 个 ERB band 的投影关系。", "把 32 个 band 的增益加权求和，得到该频点的 gain。", "实部、虚部同时乘上 gain，得到 masked spectrum。"],
    note: "这是 CPU 频谱循环；它没有启动 GPU thread。"
  },
  deep_filter: {
    title: "低频 Deep Filter：复数历史加权",
    formula: "S_df[k] = Σ(j=0..4) C[k,j] × S_raw[t-j,k]\n(a+ib)(c+id) = (ac-bd) + i(ad+bc)",
    steps: ["取当前原始频谱和 DF history 中的历史频点。", "每个 tap 做一次复数乘法。", "把 5 个 tap 的结果相加，得到低频增强频谱。", "高于 DF bins 的频点沿用 ERB mask 结果。"],
    note: "关键点：低频 DF 读取原始频谱；高频才从 mask 结果补齐。"
  },
  synthesis: {
    title: "ISTFT：从频谱回到波形",
    formula: "x_frame[n] = Σ(k=0..319) Re(S[k] × e^(i·2πkn/320)) × synthesisWindow[n]\noutput = currentFrame + overlapHistory",
    steps: ["把正频率和镜像负频率组合成实数波形。", "对每个时间采样点，把所有频点贡献相加。", "乘合成窗，并和上一窗口的重叠部分相加。", "只输出当前 hop 的 160 点，剩余部分留给下一帧。"],
    note: "当前完整 Hush session 使用 CPU 参考 overlap-add；Metal ISTFT shader 是可用原语，但没有被这条 session 调用。"
  },
  state: {
    title: "状态：让下一帧记得上一帧",
    formula: "state_next = {\n  GRU hidden,\n  causal Pad history,\n  DF spectrum history,\n  synthesis overlap history\n}",
    steps: ["Encoder / decoder 的 GRU 保存最后 hidden。", "因果 Pad 保存卷积下一帧需要的历史。", "DF 保存低频复数频谱，合成器保存 overlap-add 尾部。", "这些状态不能指向会被 pool.reset() 覆盖的临时内存。"],
    note: "流式模型的“连续感”来自状态；每一帧从零开始会让结果和实时处理不一样。"
  },
  output: {
    title: "输出：把帧重新接成声音",
    formula: "T frames × 160 samples/frame = T×160 samples\nsample_rate = 16000 · channels = 1",
    steps: ["每个输出 AudioFrame 都保持 160 个采样点。", "批次输出按原顺序追加到 WAV 或播放缓冲。", "播放速度由采样率决定，不由推理完成速度决定。"],
    note: "推理很快或很慢都不能改变 WAV 的采样率；时长错乱优先检查帧数量和采样率。"
  }
};
