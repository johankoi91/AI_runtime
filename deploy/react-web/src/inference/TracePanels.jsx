import { Badge, Facts, UiButton } from "../components/ui.jsx";
import data from "../lib/trace-data.js";
import { calculations, measuredPhases, phases } from "./phase-data.js";

export function ms(value) {
  return value === null || value === undefined ? "未采集" : Number(value).toFixed(3) + " ms";
}

export function deviceLabel(values) {
  return values.length ? values.map(value => value === "metal" ? "Metal" : value === "cpu" ? "CPU" : value).join(" + ") : "未采集";
}

export function Overview({ state, events }) {
  const real = state.mode === "real";
  const batch = state.batch;
  const coverage = measuredPhases.filter(phase => data.stageStats(events, phase).count > 0).length;
  const metrics = real ? [
    ["当前音频范围", batch.frames ? `${batch.frames} 帧 / 次调用` : "帧数未采集", batch.frameStart === null ? "未收到帧号" : `从帧 ${batch.frameStart} 开始 · ${batch.frames === 1 ? "逐帧" : "批处理"}`],
    ["本次推理总耗时", ms(batch.duration), batch.error ? "本次调用发生错误" : batch.incomplete ? "缺少开始或结束事件" : "本次调用主机墙钟耗时"],
    ["音频时长 / 处理模式", batch.frames ? `${batch.frames * 10} ms` : "未采集", `按当前 16 kHz / 160 hop 契约换算 · ${batch.frames > 1 ? "非单帧延迟" : "不含设备采集/播放缓冲"}`],
    ["观测覆盖", `${coverage} / ${measuredPhases.length} 阶段`, batch.incomplete || coverage < measuredPhases.length ? "采样 / 丢弃 / 尚未上传，缺失不补造" : "只统计已收到的阶段完成记录"]
  ] : [
    ["当前音频范围", "160 samples", "源码契约 · 16 kHz mono"],
    ["本次推理总耗时", "—", "讲解模式不生成实测耗时"],
    ["音频时长 / 处理模式", "10 ms", "音频帧时长不等于推理延迟"],
    ["观测覆盖", "讲解模式", "用源码解释实际调用关系"]
  ];
  const ids = ["frame", "latency", "audio", "coverage"];
  return <section className="metrics" aria-label="本次推理概览">
    {metrics.map(([title, value, hint], index) => <div className="metric" key={ids[index]}>
      <small>{title}</small><strong id={`${ids[index]}Metric`}>{value}</strong><span id={`${ids[index]}Hint`}>{hint}</span>
    </div>)}
  </section>;
}

const graphLabels = {
  frontend: ["① 前端特征", "Vorbis · STFT · ERB / DF"],
  encoder: ["② Encoder", "enc.onnx"],
  erb: ["③ ERB decoder", "erb_dec.onnx → m"],
  df: ["④ DF decoder", "df_dec.onnx → coefs"],
  mask: ["⑤ ERB mask", "频带增益 → 频点"],
  deep_filter: ["⑥ Deep filter", "低频原始频谱 + 历史"],
  synthesis: ["⑦ ISTFT / OLA", "逆变换 · overlap-add"]
};

export function GraphNode({ phase, selected, real, events, onSelect }) {
  const stats = data.stageStats(events, phase);
  const classes = ["graph-node", selected === phase && "active", real && stats.last && "observed", real && !stats.last && "unseen", stats.error && "failed"].filter(Boolean).join(" ");
  const deviceClass = stats.devices.includes("metal") ? stats.devices.length > 1 ? "mixed" : "metal" : "";
  return <UiButton className={classes} data-phase={phase} id={`flow-${phase}`} aria-pressed={selected === phase} onClick={() => onSelect(phase)}>
    <span className={`device ${deviceClass}`}>{real ? deviceLabel(stats.devices) : phases[phase].device + " · 源码"}</span>
    <b>{graphLabels[phase][0]}</b><small>{graphLabels[phase][1]}</small>
    <span className="node-bottom">{real
      ? stats.count ? `${ms(stats.duration)} · ${stats.count} 次完成` : stats.last ? "已开始 / 完成事件未采集" : "未采集"
      : phases[phase].output}</span>
  </UiButton>;
}

export function Inspector({ phase, state, events, observation }) {
  const meta = phases[phase];
  const real = state.mode === "real";
  const stats = data.stageStats(events, phase);
  const values = data.attributes(observation?.detail || "");
  const evidence = ["shape", "spectrum", "erb", "df", "samples"].filter(key => values[key] !== undefined).map(key => `${key}=${values[key]}`);
  const cards = [
    ["输入 · 源码契约", meta.input, "解释数据含义；不作为本次运行的 shape 实测值", true],
    ["输出 · 源码契约", meta.output, "T 为一次调用中的实际帧数", true]
  ];
  if (real) cards.unshift(["本次事件 · 实测记录", evidence.join(" · ") || "未上报 shape", observation ? observation.detail || "此事件没有详细字段" : "该阶段无事件；保留源码契约供理解", false]);
  const rows = real ? [
    ["事件", observation?.event || "未采集"],
    ["相对本次调用", observation ? `+${Math.max(0, data.timeMs(observation) - state.batch.originMs).toFixed(3)} ms` : "未采集"],
    ["该事件耗时", observation && /_complete$|_error$/.test(observation.event) ? ms(observation.duration_ms) : "未完成 / 非计时事件"],
    ["该事件字节数", observation ? `${observation.bytes} B` : "未采集"],
    ["节点 / 算子", values.node ? `${values.node} · ${values.op || ""}` : "阶段级事件"]
  ] : [["来源", "源码机制说明"], ["实测耗时", "未加载真实运行"]];
  const timingStats = measuredPhases.map(key => data.stageStats(events, key));
  const maximum = Math.max(...timingStats.map(item => item.duration || 0), 0.001);
  const timingNames = ["前端", "Encoder", "ERB decoder", "DF decoder", "Mask", "Deep filter", "ISTFT / OLA"];
  return <section className="workspace">
    <article className="panel">
      <div className="section-title"><h2 id="stageTitle">{meta.title}</h2><Badge id="deviceBadge" kind="lesson">{real ? deviceLabel(values.device ? [values.device] : stats.devices) : meta.device + " · 源码"}</Badge></div>
      <p className="explanation" id="stageExplanation">{meta.explanation}</p>
      <Facts id="eventFacts" rows={rows} />
      <div className="callout" id="stageNote">{meta.note}{real && !stats.last ? " 当前批次没有该阶段事件，不能由此判断它没有执行。" : ""}</div>
      <p className="muted" id="sourcePath" style={{ marginTop: 12 }}>{meta.source}</p>
    </article>
    <article className="panel">
      <div className="section-title"><h2>数据怎样变化</h2><Badge id="shapeBadge">{real && observation ? "契约 + 事件证据" : "源码契约"}</Badge></div>
      <div id="tensorCards">{cards.map(([title, value, hint, contract]) => <div className={`evidence ${contract ? "contract" : ""}`} key={title}>
        <div className="label"><span>{title}</span></div><code>{value}</code><small>{hint}</small>
      </div>)}</div>
      <p className="muted">只采集 shape、字节数和时序。波形、Tensor 数值及人声内容不上传。</p>
    </article>
    <article className="panel">
      <div className="section-title"><h2>阶段耗时</h2><Badge id="timingBadge">{real ? "本批已采集累计" : "无实测值"}</Badge></div>
      <div id="stageTimings">{timingStats.map((item, index) => {
        const value = real ? item.duration : null;
        return <div className="timing-row" key={measuredPhases[index]}><span>{timingNames[index]}</span><div className="bar"><i style={{ width: (value === null ? 0 : value / maximum * 100) + "%" }} /></div><span>{ms(value)}</span></div>;
      })}</div>
      <p className="muted" style={{ marginTop: 10 }}>阶段图统计已收到的完成事件；批次总耗时独立记录。缺失或采样事件不补零，不用部分耗时相加冒充整批耗时。</p>
    </article>
  </section>;
}

export function Mechanics({ phase, real, events }) {
  const spec = calculations[phase] || calculations.input;
  const nodes = data.stageStats(events, phase).nodes;
  const metalEvents = events.filter(event => event.stage === "metal");
  const devices = nodes.map(event => data.attributes(event.detail).device).filter(Boolean);
  const hasMetal = devices.includes("metal") || (["encoder", "erb", "df"].includes(phase) && metalEvents.length > 0);
  const observed = real ? `实测记录：节点 ${nodes.length} 个 · Metal 底层事件 ${metalEvents.length} 个 · 设备 ${deviceLabel([...new Set(devices)])}` : "这是源码级计算模型";
  const dispatch = metalEvents.find(event => event.event === "dispatch_configured");
  const dispatchText = dispatch ? dispatch.detail : "本批没有单独上报 dispatch grid";
  const lane = hasMetal ? [["1", "Tensor", "on"], ["2", "MTLBuffer", "on"], ["3", "Command", "on"], ["4", "Threads", "on"], ["5", "Copy back", "on"]]
    : [["1", "Tensor", "cpu"], ["2", "循环", "cpu"], ["3", "写 output", "cpu"], ["4", "更新 state", "cpu"], ["5", "下一阶段", "cpu"]];
  const explanation = hasMetal
    ? "实际含 Metal 节点：每个输出位置通常由一个概念线程负责；线程从输入 / 权重 buffer 读取，计算后写输出 buffer。MetalBackend 会创建 command buffer、配置 encoder、dispatch grid，完成后同步并回读 Tensor。"
    : "当前阶段实际走 CPU：没有 Metal thread。可以把它想成一个或多个 CPU 循环逐个读取 Tensor 元素，计算后写入 MemoryPool 的 output，再交给下一个阶段。";
  return <section className="panel mechanics">
    <div className="section-title"><h2>计算过程 / GPU 是怎么工作的</h2><Badge id="mechanicsBadge" kind={real ? "live" : "lesson"}>{real ? "源码模型 + 实测证据" : "源码模型"}</Badge></div>
    <p className="muted" id="mechanicsIntro">这里把当前阶段翻译成小白可以跟着算的公式，再说明它如何映射到 CPU 循环或 Metal 线程。公式用于理解机制；实际数值只来自真实事件中的 shape、bytes 和耗时。</p>
    <div className="mechanics-grid">
      <article className="mechanics-card" id="mathCard"><h3>① 公式：数据到底怎么算</h3><div className="formula">{spec.formula}</div>
        {spec.steps.map((step, index) => <div className="mechanics-step" key={step}><i>{index + 1}</i><div>{step}</div></div>)}<p className="mechanics-note">{spec.note}</p>
      </article>
      <article className="mechanics-card" id="gpuCard"><h3>② 执行：CPU / GPU 怎么跑</h3>
        <div className="gpu-lane">{lane.map(([number, label, kind]) => <div className={`gpu-box ${kind}`} key={number}><b>{number}</b><br />{label}</div>)}</div>
        <div className="mechanics-step"><i>↳</i><div>{explanation}</div></div>
        <div className="mechanics-step"><i>∥</i><div><b>线程如何分工：</b>卷积 / MatMul 常见映射是“一个输出元素 = 一个线程”；线程内部仍要循环累加输入和权重。ERB filterbank 则常见为“一个 frame × band = 一个线程”。</div></div>
        <div className="mechanics-step"><i>↔</i><div><b>同步在哪里：</b>本 runtime 当前 Metal 算子会等待完成并把结果复制回 Tensor，所以跨算子仍有同步点；这也是小算子吞吐受影响的原因。</div></div>
        <div className={`observed-line ${real ? "" : "warn"}`}>{real ? `实测：${observed}。本批最新 dispatch 记录：${dispatchText}` : "源码模型：页面没有真实事件，所以不展示伪造的 GPU grid / 耗时。"}</div>
      </article>
    </div>
  </section>;
}

export function Details({ phase, events, batch, dispatch }) {
  const stats = data.stageStats(events, phase);
  const retained = events.filter(event => event.event === "state_retained").pop();
  const values = data.attributes(retained?.detail || "");
  return <details className="panel secondary" id="eventDetails">
    <summary>展开辅助信息：节点详情 / 原始事件 / 观测边界</summary>
    <div className="detail-grid">
      <section><h3>当前阶段已采集的节点</h3><div className="table-scroll" id="nodeTable">
        {stats.nodes.length ? <table><thead><tr><th>节点 / 算子</th><th>设备</th><th>耗时</th></tr></thead><tbody>
          {stats.nodes.slice(0, 100).map(event => {
            const fields = data.attributes(event.detail);
            return <tr key={event.sequence}><td><UiButton data-sequence={event.sequence} onClick={() => dispatch({ type: "node", sequence: event.sequence })}>{`${fields.op || ""} · ${fields.node || ""}`}</UiButton></td><td>{fields.device || "未采集"}</td><td>{ms(event.duration_ms)}</td></tr>;
          })}</tbody></table> : <div className="empty">当前阶段没有采集到节点详情；可尝试首个批次，或查看其他图阶段。</div>}
      </div></section>
      <section><h3>跨帧状态与观测边界</h3><Facts id="stateFacts" rows={[
        ["GRU 状态 Tensor 数", values.recurrent || "未采集"], ["causal Pad 状态数", values.padding || "未采集"],
        ["DF + 合成历史字节", retained ? `${retained.bytes} B` : "未采集"], ["临时池当前使用", values.pool_used_bytes ? `${values.pool_used_bytes} B` : "未采集"]
      ]} /><div className="callout">Hush 前端、频谱增强和重建当前走 CPU。ONNX 的 Conv / 激活等支持算子走 Metal，GRU 和部分形状 / 广播算子走 CPU。上报有采样与非阻塞丢弃机制；未见事件不能解释成没执行。</div></section>
    </div>
    <h3 style={{ marginTop: 18 }}>原始事件（当前帧批次，最多 200 条）</h3>
    <div className="log" id="eventLog">{events.length ? events.slice(0, 200).map(event => <div key={`${event.sequence}-${event._time}`}>{`#${event.sequence} +${Math.max(0, data.timeMs(event) - batch.originMs).toFixed(3)} ms · ${event.stage} / ${event.event} · ${event.detail || ""}`}</div>) : <div className="empty">讲解模式没有真实事件日志。</div>}</div>
  </details>;
}
