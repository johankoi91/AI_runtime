(function () {
  "use strict";
  var data = window.RuntimeTraceData;
  var $ = function (id) { return document.getElementById(id); };
  var phaseOrder = ["input", "frontend", "encoder", "erb", "df", "mask", "deep_filter", "synthesis", "state", "output"];
  var measuredPhases = ["frontend", "encoder", "erb", "df", "mask", "deep_filter", "synthesis"];
  var phases = {
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
  var state = {
    mode: "lesson", runs: [], run: null, batches: [], batch: null,
    events: [], cursor: 0, inspected: null, timer: null, loading: false,
    selectedRun: new URLSearchParams(location.search).get("run") || ""
  };
  var requestedRun = state.selectedRun;
  var loadedOnce = false;

  function escape(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (character) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character];
    });
  }
  function ms(value) { return value === null || value === undefined ? "未采集" : Number(value).toFixed(3) + " ms"; }
  function facts(rows) {
    return rows.map(function (row) { return '<div class="fact"><span>' + escape(row[0]) + '</span><b>' + escape(row[1]) + '</b></div>'; }).join("");
  }
  function badge(element, text, kind) {
    element.textContent = text;
    element.className = "badge " + (kind || "");
  }
  function currentEvent() { return state.mode === "real" ? state.events[state.cursor] || null : null; }
  function currentPhase() {
    return state.inspected || (state.mode === "real" ? data.phaseFor(currentEvent() || {}) : phaseOrder[state.cursor]) || "input";
  }
  function currentEvents() { return state.mode === "real" && state.batch ? state.batch.events : []; }
  function deviceLabel(values) {
    return values.length ? values.map(function (value) { return value === "metal" ? "Metal" : value === "cpu" ? "CPU" : value; }).join(" + ") : "未采集";
  }
  function stop() {
    if (state.timer !== null) clearInterval(state.timer);
    state.timer = null;
    $("pause").disabled = true;
    $("play").textContent = "▶ 回放流程";
  }
  function rebuildEvents(preferred) {
    state.events = state.batch ? data.milestones(state.batch.events, $("nodeDetail").checked) : [];
    var found = preferred ? state.events.findIndex(function (event) {
      return event.sequence === preferred.sequence && event._time === preferred._time;
    }) : -1;
    state.cursor = found >= 0 ? found : 0;
    state.inspected = null;
  }
  function lesson() {
    stop();
    state.mode = "lesson";
    state.cursor = 0;
    state.inspected = null;
    render();
  }
  function selectBatch(preserve) {
    var previous = currentEvent();
    state.batch = state.batches.find(function (batch) { return batch.id === $("batchSelect").value; }) || state.batches[0] || null;
    state.mode = state.batch ? "real" : "lesson";
    if (!preserve) stop();
    rebuildEvents(preserve ? previous : null);
    render();
  }
  function showRun(preserve) {
    var previousRun = state.run;
    var previousBatch = state.batch;
    state.run = state.runs.find(function (run) { return run.id === $("traceRun").value; }) || null;
    state.selectedRun = state.run ? state.run.id : "";
    var keep = preserve && previousRun && state.run && previousRun.id === state.run.id;
    if (!keep) stop();
    state.batches = state.run && state.run.kind === "hush" ? data.batches(state.run.events) : [];
    $("batchSelect").innerHTML = state.batches.length ? state.batches.map(function (batch) {
      var range = batch.frameStart === null ? "帧号未采集" : "帧 " + batch.frameStart +
        (batch.frames > 1 ? "–" + (batch.frameStart + batch.frames - 1) : "");
      var label = range + " · " + (batch.frames || "?") + " 帧 · " +
        (batch.error ? "发生错误" : batch.incomplete ? "记录不完整" : ms(batch.duration));
      return '<option value="' + escape(batch.id) + '">' + escape(label) + '</option>';
    }).join("") : '<option value="">没有 Hush 帧批次记录</option>';
    $("batchSelect").disabled = !state.batches.length;
    if (keep && previousBatch && state.batches.some(function (batch) { return batch.id === previousBatch.id; }))
      $("batchSelect").value = previousBatch.id;
    else if (state.batches.length) {
      var first = state.batches.find(function (batch) { return batch.complete && !batch.incomplete; }) || state.batches[0];
      $("batchSelect").value = first.id;
    }
    if (state.batches.length) selectBatch(keep);
    else {
      lesson();
      if (state.run) $("connectionStatus").textContent = state.run.kind === "old"
        ? "当前运行是旧版 MatMul / Add demo，没有 Hush 推理事件。仍可使用源码讲解。"
        : "当前运行没有可关联的 Hush 阶段记录。数据可能尚在上传或已采样丢弃。";
    }
  }

  function renderOverview() {
    var real = state.mode === "real";
    var batch = state.batch;
    badge($("modeBadge"), real ? "真实推理回放" : "源码讲解", real ? "live" : "lesson");
    if (!real) {
      $("frameMetric").textContent = "160 samples";
      $("frameHint").textContent = "源码契约 · 16 kHz mono";
      $("latencyMetric").textContent = "—";
      $("latencyHint").textContent = "讲解模式不生成实测耗时";
      $("audioMetric").textContent = "10 ms";
      $("audioHint").textContent = "音频帧时长不等于推理延迟";
      $("coverageMetric").textContent = "讲解模式";
      $("coverageHint").textContent = "用源码解释实际调用关系";
      $("flowHint").textContent = "点击节点查看机制；加载真实运行后，回放、Tensor、设备和耗时将同步更新。";
      return;
    }
    var coverage = measuredPhases.filter(function (phase) { return data.stageStats(batch.events, phase).count > 0; }).length;
    $("frameMetric").textContent = batch.frames ? batch.frames + " 帧 / 次调用" : "帧数未采集";
    $("frameHint").textContent = batch.frameStart === null ? "未收到帧号" : "从帧 " + batch.frameStart + " 开始 · " + (batch.frames === 1 ? "逐帧" : "批处理");
    $("latencyMetric").textContent = ms(batch.duration);
    $("latencyHint").textContent = batch.error ? "本次调用发生错误" : batch.incomplete ? "缺少开始或结束事件" : "本次调用主机墙钟耗时";
    $("audioMetric").textContent = batch.frames ? batch.frames * 10 + " ms" : "未采集";
    $("audioHint").textContent = "按当前 16 kHz / 160 hop 契约换算 · " + (batch.frames > 1 ? "非单帧延迟" : "不含设备采集/播放缓冲");
    $("coverageMetric").textContent = coverage + " / " + measuredPhases.length + " 阶段";
    $("coverageHint").textContent = batch.incomplete || coverage < measuredPhases.length ? "采样 / 丢弃 / 尚未上传，缺失不补造" : "只统计已收到的阶段完成记录";
    $("flowHint").textContent = "Run: " + state.run.id + " · " + batch.events.length +
      " 条关联事件。当前模型分发位置由真实节点记录确认；回放为慢速阅读。";
  }

  function renderGraph(phase) {
    var events = currentEvents();
    document.querySelectorAll(".graph-node").forEach(function (element) {
      var key = element.dataset.phase;
      var stats = data.stageStats(events, key);
      element.classList.toggle("active", key === phase);
      element.classList.toggle("observed", state.mode === "real" && !!stats.last);
      element.classList.toggle("unseen", state.mode === "real" && !stats.last);
      element.classList.toggle("failed", stats.error);
      element.setAttribute("aria-pressed", String(key === phase));
      var device = element.querySelector(".device");
      device.textContent = state.mode === "real" ? deviceLabel(stats.devices) : phases[key].device + " · 源码";
      device.className = "device " + (stats.devices.indexOf("metal") >= 0 ? stats.devices.length > 1 ? "mixed" : "metal" : "");
      element.querySelector(".node-bottom").textContent = state.mode === "real"
        ? stats.count ? ms(stats.duration) + " · " + stats.count + " 次完成" : stats.last ? "已开始 / 完成事件未采集" : "未采集"
        : phases[key].output;
    });
    $("flow-state").classList.toggle("active", phase === "state");
    var retained = events.filter(function (event) { return event.event === "state_retained"; }).pop();
    var metadata = retained ? data.attributes(retained.detail) : {};
    $("stateSummary").textContent = retained ? "GRU " + (metadata.recurrent || "?") + " · Pad " +
      (metadata.padding || "?") + " · 历史 " + retained.bytes + " B" : state.mode === "real" ? "状态事件未采集" : "独立于临时 MemoryPool";
  }

  function renderInspector(phase) {
    var meta = phases[phase];
    var real = state.mode === "real";
    var event = state.inspected ? null : currentEvent();
    var stats = data.stageStats(currentEvents(), phase);
    var observation = event && data.phaseFor(event) === phase ? event : stats.output || stats.last;
    var values = observation ? data.attributes(observation.detail) : {};
    $("stageTitle").textContent = meta.title;
    $("stageExplanation").textContent = meta.explanation;
    badge($("deviceBadge"), real ? deviceLabel(values.device ? [values.device] : stats.devices) : meta.device + " · 源码", "lesson");
    $("eventFacts").innerHTML = real ? facts([
      ["事件", observation ? observation.event : "未采集"],
      ["相对本次调用", observation ? "+" + Math.max(0, data.timeMs(observation) - state.batch.originMs).toFixed(3) + " ms" : "未采集"],
      ["该事件耗时", observation && /_complete$|_error$/.test(observation.event) ? ms(observation.duration_ms) : "未完成 / 非计时事件"],
      ["该事件字节数", observation ? observation.bytes + " B" : "未采集"],
      ["节点 / 算子", values.node ? values.node + " · " + (values.op || "") : "阶段级事件"]
    ]) : facts([["来源", "源码机制说明"], ["实测耗时", "未加载真实运行"]]);
    $("stageNote").textContent = meta.note + (real && !stats.last ? " 当前批次没有该阶段事件，不能由此判断它没有执行。" : "");
    $("sourcePath").textContent = meta.source;
    badge($("shapeBadge"), real && observation ? "契约 + 事件证据" : "源码契约");
    var cards = [
      ["输入 · 源码契约", meta.input, "解释数据含义；不作为本次运行的 shape 实测值", true],
      ["输出 · 源码契约", meta.output, "T 为一次调用中的实际帧数", true]
    ];
    var evidence = [];
    ["shape", "spectrum", "erb", "df", "samples"].forEach(function (key) {
      if (values[key] !== undefined) evidence.push(key + "=" + values[key]);
    });
    if (real) cards.unshift([
      "本次事件 · 实测记录",
      evidence.length ? evidence.join(" · ") : "未上报 shape",
      observation ? observation.detail || "此事件没有详细字段" : "该阶段无事件；保留源码契约供理解", false
    ]);
    $("tensorCards").innerHTML = cards.map(function (card) {
      return '<div class="evidence ' + (card[3] ? "contract" : "") + '"><div class="label"><span>' +
        escape(card[0]) + '</span></div><code>' + escape(card[1]) + '</code><small>' + escape(card[2]) + '</small></div>';
    }).join("");
    badge($("eventBadge"), real ? observation ? observation.event : "未采集" : "源码讲解", real ? "live" : "lesson");
  }

  var calculations = {
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

  function renderMechanics(phase) {
    var spec = calculations[phase] || calculations.input;
    var real = state.mode === "real";
    var events = currentEvents();
    var stats = data.stageStats(events, phase);
    var nodeEvents = stats.nodes;
    var metalEvents = events.filter(function (event) { return event.stage === "metal"; });
    var deviceValues = nodeEvents.map(function (event) { return data.attributes(event.detail).device; }).filter(Boolean);
    var metalPhase = phase === "encoder" || phase === "erb" || phase === "df";
    var hasMetal = deviceValues.indexOf("metal") >= 0 || (metalPhase && metalEvents.length > 0);
    var hasCpu = deviceValues.indexOf("cpu") >= 0 || (real && (phase === "frontend" || phase === "mask" ||
      phase === "deep_filter" || phase === "synthesis" || phase === "state"));
    var observed = real ? "实测记录：节点 " + nodeEvents.length + " 个 · Metal 底层事件 " +
      metalEvents.length + " 个 · 设备 " + deviceLabel(Array.from(new Set(deviceValues))) : "这是源码级计算模型";
    var dispatch = metalEvents.find(function (event) { return event.event === "dispatch_configured"; });
    var dispatchText = dispatch ? dispatch.detail : "本批没有单独上报 dispatch grid";
    $("mathCard").innerHTML = "<h3>① 公式：数据到底怎么算</h3>" +
      "<div class=\"formula\">" + escape(spec.formula) + "</div>" +
      spec.steps.map(function (step, index) {
        return '<div class="mechanics-step"><i>' + (index + 1) + '</i><div>' + escape(step) + '</div></div>';
      }).join("") +
      '<p class="mechanics-note">' + escape(spec.note) + "</p>";
    var lane = hasMetal ? [
      ["1", "Tensor", "on"],
      ["2", "MTLBuffer", "on"],
      ["3", "Command", "on"],
      ["4", "Threads", "on"],
      ["5", "Copy back", "on"]
    ] : [
      ["1", "Tensor", "cpu"],
      ["2", "循环", "cpu"],
      ["3", "写 output", "cpu"],
      ["4", "更新 state", "cpu"],
      ["5", "下一阶段", "cpu"]
    ];
    var gpuExplanation = hasMetal
      ? "实际含 Metal 节点：每个输出位置通常由一个概念线程负责；线程从输入 / 权重 buffer 读取，计算后写输出 buffer。MetalBackend 会创建 command buffer、配置 encoder、dispatch grid，完成后同步并回读 Tensor。"
      : "当前阶段实际走 CPU：没有 Metal thread。可以把它想成一个或多个 CPU 循环逐个读取 Tensor 元素，计算后写入 MemoryPool 的 output，再交给下一个阶段。";
    var actual = real
      ? "实测：" + escape(observed) + "。本批最新 dispatch 记录：" + escape(dispatchText)
      : "源码模型：页面没有真实事件，所以不展示伪造的 GPU grid / 耗时。";
    $("gpuCard").innerHTML = "<h3>② 执行：CPU / GPU 怎么跑</h3>" +
      '<div class="gpu-lane">' + lane.map(function (item) {
        return '<div class="gpu-box ' + item[2] + '"><b>' + item[0] + '</b><br>' + item[1] + '</div>';
      }).join("") + "</div>" +
      '<div class="mechanics-step"><i>↳</i><div>' + escape(gpuExplanation) + "</div></div>" +
      '<div class="mechanics-step"><i>∥</i><div><b>线程如何分工：</b>卷积 / MatMul 常见映射是“一个输出元素 = 一个线程”；线程内部仍要循环累加输入和权重。ERB filterbank 则常见为“一个 frame × band = 一个线程”。</div></div>' +
      '<div class="mechanics-step"><i>↔</i><div><b>同步在哪里：</b>本 runtime 当前 Metal 算子会等待完成并把结果复制回 Tensor，所以跨算子仍有同步点；这也是小算子吞吐受影响的原因。</div></div>' +
      '<div class="observed-line ' + (real ? "" : "warn") + '">' + actual + "</div>";
    badge($("mechanicsBadge"), real ? "源码模型 + 实测证据" : "源码模型", real ? "live" : "lesson");
  }

  function renderTimings() {
    var real = state.mode === "real";
    var stats = measuredPhases.map(function (phase) { return { phase: phase, value: data.stageStats(currentEvents(), phase) }; });
    var maximum = Math.max.apply(null, stats.map(function (item) { return item.value.duration || 0; }).concat([.001]));
    var names = ["前端", "Encoder", "ERB decoder", "DF decoder", "Mask", "Deep filter", "ISTFT / OLA"];
    $("stageTimings").innerHTML = stats.map(function (item, index) {
      var value = real ? item.value.duration : null;
      return '<div class="timing-row"><span>' + escape(names[index]) + '</span><div class="bar"><i style="width:' +
        (value === null ? 0 : value / maximum * 100) + '%"></i></div><span>' + escape(ms(value)) + '</span></div>';
    }).join("");
    badge($("timingBadge"), real ? "本批已采集累计" : "无实测值");
  }

  function renderDetails(phase) {
    var stats = data.stageStats(currentEvents(), phase);
    $("nodeTable").innerHTML = stats.nodes.length ? '<table><thead><tr><th>节点 / 算子</th><th>设备</th><th>耗时</th></tr></thead><tbody>' +
      stats.nodes.slice(0, 100).map(function (event) {
        var fields = data.attributes(event.detail);
        return '<tr><td><button data-sequence="' + event.sequence + '">' + escape((fields.op || "") + " · " +
          (fields.node || "")) + '</button></td><td>' + escape(fields.device || "未采集") + '</td><td>' + ms(event.duration_ms) + '</td></tr>';
      }).join("") + "</tbody></table>" : '<div class="empty">当前阶段没有采集到节点详情；可尝试首个批次，或查看其他图阶段。</div>';
    $("nodeTable").querySelectorAll("button").forEach(function (button) {
      button.onclick = function () {
        var selected = currentEvents().find(function (event) { return event.sequence === Number(button.dataset.sequence); });
        stop();
        $("nodeDetail").checked = true;
        rebuildEvents(selected);
        render();
      };
    });
    var retained = currentEvents().filter(function (event) { return event.event === "state_retained"; }).pop();
    var values = data.attributes(retained ? retained.detail : "");
    $("stateFacts").innerHTML = facts([
      ["GRU 状态 Tensor 数", values.recurrent || "未采集"],
      ["causal Pad 状态数", values.padding || "未采集"],
      ["DF + 合成历史字节", retained ? retained.bytes + " B" : "未采集"],
      ["临时池当前使用", values.pool_used_bytes ? values.pool_used_bytes + " B" : "未采集"]
    ]);
    $("eventLog").innerHTML = currentEvents().slice(0, 200).map(function (event) {
      return "<div>#" + event.sequence + " +" + Math.max(0, data.timeMs(event) - state.batch.originMs).toFixed(3) +
        " ms · " + escape(event.stage + " / " + event.event) + " · " + escape(event.detail || "") + "</div>";
    }).join("") || '<div class="empty">讲解模式没有真实事件日志。</div>';
  }

  function render() {
    var phase = currentPhase();
    var count = state.mode === "real" ? state.events.length : phaseOrder.length;
    renderOverview();
    renderGraph(phase);
    renderInspector(phase);
    renderMechanics(phase);
    renderTimings();
    renderDetails(phase);
    $("position").textContent = count ? (state.cursor + 1) + " / " + count : "没有可回放事件";
    $("seek").max = Math.max(0, count - 1);
    $("seek").value = state.cursor;
    $("seek").disabled = !count;
    $("prev").disabled = !count || state.cursor === 0;
    $("next").disabled = !count || state.cursor >= count - 1;
    $("play").disabled = !count;
    var event = currentEvent();
    $("elapsed").textContent = event ? "+" + Math.max(0, data.timeMs(event) - state.batch.originMs).toFixed(3) + " ms" : "源码讲解";
  }
  function move(delta) {
    stop();
    var count = state.mode === "real" ? state.events.length : phaseOrder.length;
    state.cursor = Math.max(0, Math.min(count - 1, state.cursor + delta));
    state.inspected = null;
    render();
  }
  function play() {
    stop();
    var count = state.mode === "real" ? state.events.length : phaseOrder.length;
    if (!count) return;
    if (state.cursor >= count - 1) state.cursor = 0;
    state.inspected = null;
    $("pause").disabled = false;
    $("play").textContent = "正在回放…";
    render();
    state.timer = setInterval(function () {
      if (state.cursor >= count - 1) { stop(); return; }
      state.cursor += 1;
      render();
    }, Number($("speed").value));
  }
  function inspectPhase(phase) {
    stop();
    if (state.mode === "lesson") state.cursor = phaseOrder.indexOf(phase);
    else {
      var index = state.events.findIndex(function (event) { return data.phaseFor(event) === phase && event.event === "stage_complete"; });
      if (index < 0) index = state.events.findIndex(function (event) { return data.phaseFor(event) === phase; });
      if (index >= 0) { state.cursor = index; state.inspected = null; }
      else state.inspected = phase;
    }
    render();
  }

  function loadTrace(manual) {
    if (state.loading) return;
    state.loading = true;
    $("reloadTrace").disabled = true;
    $("connectionStatus").textContent = "正在读取最近 24 小时的真实事件…";
    var query = 'from(bucket: ' + JSON.stringify($("bucket").value) + ') |> range(start: -24h)' +
      ' |> filter(fn: (r) => r._measurement == "runtime_trace")' +
      (requestedRun ? ' |> filter(fn: (r) => r.run_id == ' + JSON.stringify(requestedRun) + ')' : "") +
      ' |> group(columns: ["_field"]) |> sort(columns: ["_time"], desc: true) |> limit(n: 20000)';
    var controller = new AbortController();
    var timeout = setTimeout(function () { controller.abort(); }, 15000);
    fetch($("api").value.replace(/\/$/, "") + "/api/v2/query?org=" + encodeURIComponent($("org").value), {
      method: "POST", signal: controller.signal,
      headers: { Authorization: "Token " + $("token").value, "Content-Type": "application/vnd.flux", Accept: "application/csv" },
      body: query
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.text();
    }).then(function (csv) {
      var events = data.parseCsv(csv);
      state.runs = data.runs(events);
      var oldRun = state.selectedRun;
      $("traceRun").innerHTML = state.runs.length ? state.runs.map(function (run) {
        return '<option value="' + escape(run.id) + '">' + escape(run.id + (run.kind === "hush" ? " · Hush" : run.kind === "old" ? " · 旧 demo" : " · 来源未识别")) + "</option>";
      }).join("") : '<option value="">暂无事件</option>';
      if (state.runs.some(function (run) { return run.id === oldRun; })) $("traceRun").value = oldRun;
      else if (state.runs.some(function (run) { return run.kind === "hush"; }))
        $("traceRun").value = state.runs.find(function (run) { return run.kind === "hush"; }).id;
      var preserveLesson = loadedOnce && state.mode === "lesson" && !manual;
      showRun(!manual);
      if (preserveLesson) lesson();
      loadedOnce = true;
      $("connectionStatus").textContent = state.run
        ? state.run.id + " · " + state.run.events.length + " 条事件 · " + state.batches.length +
          " 个已观测调用 · " + (state.run.events.some(function (event) { return event.event === "run_error"; })
            ? "运行记录含错误" : state.run.events.some(function (event) { return event.event === "run_complete"; }) ? "测试已结束" : "采集中 / 等待结束事件")
        : "最近 24 小时暂无事件。可在 App 运行测试，或先使用源码讲解。";
    }).catch(function (error) {
      $("connectionStatus").textContent = "读取失败（" + error.message + "）。" +
        (state.run ? "保留上次已加载运行；不会把旧数据当成新结果。" : "当前仍可使用源码讲解。");
    }).finally(function () { clearTimeout(timeout); state.loading = false; $("reloadTrace").disabled = false; });
  }
  $("play").onclick = play;
  $("pause").onclick = stop;
  $("prev").onclick = function () { move(-1); };
  $("next").onclick = function () { move(1); };
  $("reset").onclick = function () { stop(); state.cursor = 0; state.inspected = null; render(); };
  $("seek").oninput = function () { stop(); state.cursor = Number($("seek").value); state.inspected = null; render(); };
  $("speed").onchange = function () { if (state.timer !== null) play(); };
  $("nodeDetail").onchange = function () { stop(); rebuildEvents(currentEvent()); render(); };
  $("lesson").onclick = lesson;
  $("traceRun").onchange = function () { showRun(false); };
  $("batchSelect").onchange = function () { selectBatch(false); };
  $("reloadTrace").onclick = function () { loadTrace(true); };
  $("config").onsubmit = function (event) { event.preventDefault(); loadTrace(true); };
  document.querySelectorAll("[data-phase]").forEach(function (element) {
    element.onclick = function () { inspectPhase(element.dataset.phase); };
  });
  $("token").value = localStorage.getItem("influx-token") || $("token").value;
  $("token").onchange = function () { localStorage.setItem("influx-token", $("token").value); };
  $("autoTrace").checked = !!requestedRun;
  setInterval(function () { if ($("autoTrace").checked && !document.hidden) loadTrace(false); }, 10000);
  render();
  loadTrace(false);
}());
