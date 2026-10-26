import { UiButton, Badge } from "../components/ui.jsx";
import data from "../lib/trace-data.js";
import { phaseOrder } from "./phase-data.js";
import { currentEvent, currentEvents } from "./trace-state.js";
import { useTrace } from "./use-trace.js";
import { Details, GraphNode, Inspector, Mechanics, Overview, ms } from "./TracePanels.jsx";

export function TracePage() {
  const { state, dispatch, config, updateConfig, loadTrace, autoTrace, setAutoTrace } = useTrace();
  const real = state.mode === "real";
  const events = currentEvents(state);
  const event = currentEvent(state);
  const phase = state.inspected || (real ? data.phaseFor(event || {}) : phaseOrder[state.cursor]) || "input";
  const stats = data.stageStats(events, phase);
  const observation = !state.inspected && event && data.phaseFor(event) === phase ? event : stats.output || stats.last;
  const count = real ? state.events.length : phaseOrder.length;
  const retained = events.filter(item => item.event === "state_retained").pop();
  const metadata = data.attributes(retained?.detail || "");
  const inspect = selected => dispatch({ type: "inspect", phase: selected });
  const node = key => <GraphNode phase={key} selected={phase} real={real} events={events} onSelect={inspect} />;
  const arrow = <span className="arrow" aria-hidden="true">→</span>;
  const flowHint = real ? `Run: ${state.run.id} · ${state.batch.events.length} 条关联事件。当前模型分发位置由真实节点记录确认；回放为慢速阅读。`
    : "点击节点查看机制；加载真实运行后，回放、Tensor、设备和耗时将同步更新。";

  return <main className="app">
    <header className="header">
      <div><div className="eyebrow">HUSH / RUNTIME INSPECTOR</div><h1>看懂一次真实推理</h1><p>声音如何进入模型，数据怎样经过 CPU / Metal，又如何回到连续音频。</p></div>
      <Badge id="modeBadge" kind={real ? "live" : "lesson"}>{real ? "真实推理回放" : "源码讲解"}</Badge>
    </header>
    <section className="panel connection">
      <div className="toolbar">
        <label className="field run-field">选择一次测试
          <select id="traceRun" value={state.selectedRun} onChange={event => dispatch({ type: "run", id: event.target.value })}>
            {state.runs.length ? state.runs.map(run => <option value={run.id} key={run.id}>{run.id + (run.kind === "hush" ? " · Hush" : run.kind === "old" ? " · 旧 demo" : " · 来源未识别")}</option>) : <option value="">{state.loaded ? "暂无事件" : "正在读取最近运行…"}</option>}
          </select>
        </label>
        <label className="field batch-field">查看帧 / 批次
          <select id="batchSelect" value={state.batch?.id || ""} disabled={!state.batches.length} onChange={event => dispatch({ type: "batch", id: event.target.value })}>
            {state.batches.length ? state.batches.map(batch => {
              const range = batch.frameStart === null ? "帧号未采集" : `帧 ${batch.frameStart}${batch.frames > 1 ? "–" + (batch.frameStart + batch.frames - 1) : ""}`;
              return <option value={batch.id} key={batch.id}>{`${range} · ${batch.frames || "?"} 帧 · ${batch.error ? "发生错误" : batch.incomplete ? "记录不完整" : ms(batch.duration)}`}</option>;
            }) : <option value="">{state.loaded ? "没有 Hush 帧批次记录" : "等待真实事件"}</option>}
          </select>
        </label>
        <UiButton id="reloadTrace" disabled={state.loading} onClick={() => loadTrace(true)}>刷新事件</UiButton>
        <label className="check"><input type="checkbox" id="autoTrace" checked={autoTrace} onChange={event => setAutoTrace(event.target.checked)} />每 10 秒刷新</label>
        <UiButton id="lesson" onClick={() => dispatch({ type: "lesson" })}>源码讲解</UiButton>
        <a className="tutorial-link" href="./tutorial.html">📚 数学与 GPU 动画教程</a>
      </div>
      <p className="status" id="connectionStatus" role="status">{state.status}</p>
      <details><summary>数据连接设置</summary>
        <form className="config" id="config" onSubmit={event => { event.preventDefault(); loadTrace(true); }}>
          <label className="field">Influx API<input id="api" value={config.api} onChange={event => updateConfig("api", event.target.value)} /></label>
          <label className="field">组织<input id="org" value={config.org} onChange={event => updateConfig("org", event.target.value)} /></label>
          <label className="field">Bucket<input id="bucket" value={config.bucket} onChange={event => updateConfig("bucket", event.target.value)} /></label>
          <label className="field">Token<input id="token" value={config.token} type="password" autoComplete="off" onChange={event => updateConfig("token", event.target.value)} /></label>
          <UiButton type="submit">连接并加载</UiButton>
        </form>
      </details>
    </section>
    <Overview state={state} events={events} />
    <section className="panel" id="realTrace">
      <div className="section-title"><h2>音频 → 模型 → 音频</h2><Badge id="eventBadge" kind={real ? "live" : "lesson"}>{real ? observation?.event || "未采集" : "源码讲解"}</Badge></div>
      <div className="pipeline-title"><p id="flowHint">{flowHint}</p></div>
      <div className="io-strip"><UiButton data-phase="input" onClick={() => inspect("input")}>输入 · Mic / WAV → 16 kHz mono → 160 samples / frame</UiButton><UiButton data-phase="output" onClick={() => inspect("output")}>输出 · AudioFrame → WAV / 播放</UiButton></div>
      <div className="pipeline" style={{ marginTop: 12 }}>
        {node("frontend")}{arrow}{node("encoder")}{arrow}<div className="branch">{node("erb")}{node("df")}</div>{arrow}<div className="branch">{node("mask")}{node("deep_filter")}</div>{arrow}{node("synthesis")}
      </div>
      <UiButton className={`state-strip${phase === "state" ? " active" : ""}`} id="flow-state" data-phase="state" style={{ width: "100%", textAlign: "left" }} onClick={() => inspect("state")}>
        <span>↶ 跨帧状态 · 前端归一化 / GRU / causal Pad / DF / 合成历史</span><span id="stateSummary">{retained ? `GRU ${metadata.recurrent || "?"} · Pad ${metadata.padding || "?"} · 历史 ${retained.bytes} B` : real ? "状态事件未采集" : "独立于临时 MemoryPool"}</span>
      </UiButton>
      <p className="muted" style={{ marginTop: 9 }}>两张 decoder 共享 Encoder 输出，当前按 ERB → DF 顺序执行。低频 DF 使用原始频谱；高频取 ERB mask 结果。状态在各阶段更新后交给下一帧。</p>
      <div className="player">
        <UiButton className="primary" id="play" disabled={!count} onClick={() => dispatch({ type: "play" })}>{state.playing ? "正在回放…" : "▶ 回放流程"}</UiButton>
        <UiButton id="pause" disabled={!state.playing} onClick={() => dispatch({ type: "pause" })}>暂停</UiButton>
        <UiButton id="prev" disabled={!count || state.cursor === 0} onClick={() => dispatch({ type: "seek", index: state.cursor - 1 })}>← 上一步</UiButton>
        <UiButton id="next" disabled={!count || state.cursor >= count - 1} onClick={() => dispatch({ type: "seek", index: state.cursor + 1 })}>下一步 →</UiButton>
        <UiButton id="reset" onClick={() => dispatch({ type: "seek", index: 0 })}>从头</UiButton>
        <label className="check"><input type="checkbox" id="nodeDetail" checked={state.nodeDetail} onChange={event => dispatch({ type: "details", value: event.target.checked })} />展开节点事件</label>
        <label className="field">回放速度<select id="speed" value={state.speed} onChange={event => dispatch({ type: "speed", value: event.target.value })}><option value="1000">慢速</option><option value="650">正常</option><option value="250">快速</option></select></label>
        <span className="position" id="position">{count ? `${state.cursor + 1} / ${count}` : "没有可回放事件"}</span>
      </div>
      <div className="seek"><span>开始</span><input id="seek" type="range" min="0" max={Math.max(0, count - 1)} value={state.cursor} disabled={!count} aria-label="推理事件进度" onChange={event => dispatch({ type: "seek", index: Number(event.target.value) })} /><span id="elapsed">{event ? `+${Math.max(0, data.timeMs(event) - state.batch.originMs).toFixed(3)} ms` : "源码讲解"}</span></div>
      <p className="muted" style={{ marginTop: 8 }}>回放会放慢事件便于阅读；旁边的 ms 保留真实记录。前端只记录一个整体阶段，Vorbis / STFT / ERB 的内部耗时未单独采集。</p>
    </section>
    <Inspector phase={phase} state={state} events={events} observation={observation} />
    <Mechanics phase={phase} real={real} events={events} />
    <Details phase={phase} events={events} batch={state.batch} dispatch={dispatch} />
    <p className="footer">时间为主机调用墙钟耗时，包含同步和采集开销；不代表 GPU 硬件执行时长。计算数值正确、实时吞吐和主讲人保真需要分别验收。</p>
    <noscript><p className="callout">请启用 JavaScript，才能加载真实运行并回放推理流程。</p></noscript>
  </main>;
}
