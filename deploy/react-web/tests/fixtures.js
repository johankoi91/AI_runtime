export function fixtureEvents(kind = "complete") {
  if (kind === "empty") return [];
  const events = [];
  let sequence = 1;
  const add = (run, stage, event, detail = "", duration = 0, bytes = 0) => {
    const timestamp = new Date(Date.UTC(2026, 9, 26, 6, 0, 0) + sequence).toISOString();
    events.push({ _time: timestamp, run_id: run, stage, event, detail, duration_ms: duration, bytes, sequence: sequence++ });
  };
  const run = "hush-fixture-visual-comparison";
  add("metal-fixture-legacy", "metal", "kernel_completed", "matmul_kernel");
  add(run, "hush.app", "run_start", "source=fixture");
  for (const frameStart of [0, 32]) {
    const detail = `backend=metal frames=32 frame_start=${frameStart}`;
    add(run, "hush.batch", "stage_start", detail);
    for (const [stage, duration, shape] of [
      ["frontend", 1.25, "spectrum=[1,161,2] erb=[1,32] df=[2,1,64]"],
      ["encoder", 12.5, "shape=[1,32,128]"],
      ["erb_decoder", 7.5, "shape=[1,1,32,32]"],
      ["df_decoder", 9.5, "shape=[1,32,64,10]"],
      ["mask", 0.25, "shape=[1,161,2]"],
      ["deep_filter", 0.5, "shape=[1,161,2]"],
      ["synthesis", 2, "samples=5120"]
    ]) {
      if (kind === "incomplete" && ["mask", "deep_filter"].includes(stage)) continue;
      const device = ["encoder", "erb_decoder", "df_decoder"].includes(stage) ? "metal" : "cpu";
      add(run, `hush.${stage}`, "stage_start", `device=${device}`);
      if (device === "metal") {
        add(run, "metal", "dispatch_configured", "grid=[128,32] threads=[256,1]");
        add(run, `hush.${stage}`, "node_complete", `node=Conv_${frameStart}_${stage} op=Conv device=metal shape=[1,16,32,8]`, 0.25, 16384);
        add(run, `hush.${stage}`, "node_complete", `node=GRU_${frameStart}_${stage} op=GRU device=cpu shape=[1,32,128]`, 0.75, 16384);
      }
      add(run, `hush.${stage}`, "stage_complete", `device=${device} ${shape}`, duration, 20480);
    }
    if (kind !== "incomplete") add(run, "hush.batch", "stage_complete", detail, 40 + frameStart / 32, 20480);
    add(run, "hush.state", "state_retained", "recurrent=5 padding=2 pool_used_bytes=3038324", 0, 2560);
  }
  if (kind !== "incomplete") add(run, "hush.app", "run_complete", "source=fixture");
  return events;
}

export function csvFor(events) {
  const encode = value => '"' + String(value).replaceAll('"', '""') + '"';
  const rows = [",result,table,_time,_value,_field,run_id,stage"];
  events.forEach(event => {
    for (const field of ["event", "detail", "duration_ms", "bytes", "sequence"]) {
      rows.push(["", "_result", 0, event._time, event[field], field, event.run_id, event.stage].map(encode).join(","));
    }
  });
  return rows.join("\r\n");
}
