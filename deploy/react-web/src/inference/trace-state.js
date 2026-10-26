import data from "../lib/trace-data.js";
import { phaseOrder } from "./phase-data.js";

export function createTraceState(requestedRun = "") {
  return {
    runs: [], run: null, batches: [], batch: null, events: [],
    mode: "lesson", cursor: 0, inspected: null, playing: false,
    nodeDetail: false, speed: 650, selectedRun: requestedRun,
    loaded: false, loading: false, status: "没有真实运行时，也可以使用源码讲解模式。"
  };
}

export function currentEvent(state) {
  return state.mode === "real" ? state.events[state.cursor] || null : null;
}

export function currentEvents(state) {
  return state.mode === "real" && state.batch ? state.batch.events : [];
}

function withEvents(state, preferred) {
  const events = state.batch ? data.milestones(state.batch.events, state.nodeDetail) : [];
  const index = preferred ? events.findIndex(event => event.sequence === preferred.sequence && event._time === preferred._time) : -1;
  return { ...state, events, cursor: Math.max(0, index), inspected: null };
}

function lesson(state) {
  return { ...state, mode: "lesson", playing: false, cursor: 0, inspected: null };
}

function selectRun(state, selectedId, preserve) {
  const run = state.runs.find(item => item.id === selectedId) || null;
  const keep = preserve && state.run && run && state.run.id === run.id;
  const batches = run?.kind === "hush" ? data.batches(run.events) : [];
  const batch = (keep && state.batch ? batches.find(item => item.id === state.batch.id) : null)
    || batches.find(item => item.complete && !item.incomplete) || batches[0] || null;
  const next = { ...state, run, selectedRun: run?.id || "", batches, batch, playing: Boolean(keep && state.playing) };
  return batch ? withEvents({ ...next, mode: "real" }, keep ? currentEvent(state) : null) : lesson(next);
}

export function traceReducer(state, action) {
  switch (action.type) {
    case "loading":
      return { ...state, loading: true, status: "正在读取最近 24 小时的真实事件…" };
    case "loaded": {
      const preserveLesson = state.loaded && state.mode === "lesson" && !action.manual;
      const selected = action.runs.find(run => run.id === state.selectedRun)
        || action.runs.find(run => run.kind === "hush") || action.runs[0];
      let next = selectRun({ ...state, runs: action.runs, loading: false, loaded: true }, selected?.id || "", !action.manual);
      if (preserveLesson) next = lesson(next);
      const run = next.run;
      const completion = run?.events.some(event => event.event === "run_error") ? "运行记录含错误"
        : run?.events.some(event => event.event === "run_complete") ? "测试已结束" : "采集中 / 等待结束事件";
      return { ...next, status: run
        ? `${run.id} · ${run.events.length} 条事件 · ${next.batches.length} 个已观测调用 · ${completion}`
        : "最近 24 小时暂无事件。可在 App 运行测试，或先使用源码讲解。" };
    }
    case "failed":
      return { ...state, loading: false, status: `读取失败（${action.message}）。${state.run ? "保留上次已加载运行；不会把旧数据当成新结果。" : "当前仍可使用源码讲解。"}` };
    case "lesson":
      return lesson(state);
    case "run": {
      const next = selectRun(state, action.id, false);
      if (!next.batch && next.run) return { ...next, status: next.run.kind === "old"
        ? "当前运行是旧版 MatMul / Add demo，没有 Hush 推理事件。仍可使用源码讲解。"
        : "当前运行没有可关联的 Hush 阶段记录。数据可能尚在上传或已采样丢弃。" };
      return next;
    }
    case "batch": {
      const batch = state.batches.find(item => item.id === action.id) || state.batches[0] || null;
      return withEvents({ ...state, batch, mode: batch ? "real" : "lesson", playing: false });
    }
    case "inspect": {
      if (state.mode === "lesson") return { ...state, cursor: phaseOrder.indexOf(action.phase), inspected: null, playing: false };
      let index = state.events.findIndex(event => data.phaseFor(event) === action.phase && event.event === "stage_complete");
      if (index < 0) index = state.events.findIndex(event => data.phaseFor(event) === action.phase);
      return { ...state, cursor: index < 0 ? state.cursor : index, inspected: index < 0 ? action.phase : null, playing: false };
    }
    case "details":
      return withEvents({ ...state, nodeDetail: action.value, playing: false }, currentEvent(state));
    case "node": {
      const selected = currentEvents(state).find(event => event.sequence === action.sequence);
      return withEvents({ ...state, nodeDetail: true, playing: false }, selected);
    }
    case "seek": {
      const count = state.mode === "real" ? state.events.length : phaseOrder.length;
      return { ...state, cursor: Math.max(0, Math.min(Math.max(0, count - 1), action.index)), inspected: null, playing: false };
    }
    case "play": {
      const count = state.mode === "real" ? state.events.length : phaseOrder.length;
      return { ...state, playing: count > 0, cursor: state.cursor >= count - 1 ? 0 : state.cursor, inspected: null };
    }
    case "pause":
      return { ...state, playing: false };
    case "tick": {
      const count = state.mode === "real" ? state.events.length : phaseOrder.length;
      return state.cursor >= count - 1 ? { ...state, playing: false } : { ...state, cursor: state.cursor + 1 };
    }
    case "speed":
      return { ...state, speed: Number(action.value) };
    default:
      return state;
  }
}
