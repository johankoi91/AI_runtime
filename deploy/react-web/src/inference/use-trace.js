import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import data from "../lib/trace-data.js";
import { readStored, writeStored } from "../lib/storage.js";
import { createTraceState, traceReducer } from "./trace-state.js";

export function useTrace() {
  const [requestedRun] = useState(() => new URLSearchParams(location.search).get("run") || "");
  const [state, dispatch] = useReducer(traceReducer, requestedRun, createTraceState);
  const [autoTrace, setAutoTrace] = useState(Boolean(requestedRun));
  const [config, setConfig] = useState(() => ({ api: "/influx", org: "mini-runtime", bucket: "mini-runtime", token: readStored("influx-token", "mini-runtime-local-token") }));
  const requestRef = useRef(null);
  const mountedRef = useRef(false);

  const loadTrace = useCallback(async (manual = false) => {
    if (requestRef.current) return;
    const controller = new AbortController();
    requestRef.current = controller;
    dispatch({ type: "loading" });
    const timeout = setTimeout(() => controller.abort(), 15000);
    const query = `from(bucket: ${JSON.stringify(config.bucket)}) |> range(start: -24h)`
      + ' |> filter(fn: (r) => r._measurement == "runtime_trace")'
      + (requestedRun ? ` |> filter(fn: (r) => r.run_id == ${JSON.stringify(requestedRun)})` : "")
      + ' |> group(columns: ["_field"]) |> sort(columns: ["_time"], desc: true) |> limit(n: 20000)';
    try {
      const response = await fetch(`${config.api.replace(/\/$/, "")}/api/v2/query?org=${encodeURIComponent(config.org)}`, {
        method: "POST", signal: controller.signal,
        headers: { Authorization: `Token ${config.token}`, "Content-Type": "application/vnd.flux", Accept: "application/csv" }, body: query
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const runs = data.runs(data.parseCsv(await response.text()));
      if (mountedRef.current) dispatch({ type: "loaded", runs, manual });
    } catch (error) {
      if (mountedRef.current) dispatch({ type: "failed", message: error.message });
    } finally {
      clearTimeout(timeout);
      if (requestRef.current === controller) requestRef.current = null;
    }
  }, [config, requestedRun]);
  const loadRef = useRef(loadTrace);
  loadRef.current = loadTrace;

  useEffect(() => {
    mountedRef.current = true;
    loadRef.current(false);
    return () => { mountedRef.current = false; requestRef.current?.abort(); };
  }, []);

  useEffect(() => {
    if (!autoTrace) return undefined;
    const timer = setInterval(() => { if (!document.hidden) loadRef.current(false); }, 10000);
    return () => clearInterval(timer);
  }, [autoTrace]);

  useEffect(() => {
    if (!state.playing) return undefined;
    const timer = setInterval(() => dispatch({ type: "tick" }), state.speed);
    return () => clearInterval(timer);
  }, [state.playing, state.speed]);

  const updateConfig = (key, value) => {
    if (key === "token") writeStored("influx-token", value);
    setConfig(current => ({ ...current, [key]: value }));
  };
  return { state, dispatch, config, updateConfig, loadTrace, autoTrace, setAutoTrace };
}
