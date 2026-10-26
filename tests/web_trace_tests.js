const assert = require("node:assert/strict");
const { parseCsv, runs, batches, phaseFor, milestones, stageStats } = require("../deploy/web/trace_data.js");
const header = ",result,table,_time,_value,_field,run_id,stage";
const csv = [
  "#datatype,string,long,dateTime:RFC3339,string,string,string,string",
  header,
  ',_result,0,2026-10-26T03:00:00.000001Z,node_complete,event,hush-test,hush.encoder',
  ',_result,0,2026-10-26T03:00:00.000002Z,kernel_completed,event,metal-test,metal',
  "",
  header,
  ',_result,1,2026-10-26T03:00:00.000001Z,"node=GRU shape=[1,32] quote=""value""",detail,hush-test,hush.encoder',
  ',_result,1,2026-10-26T03:00:00.000002Z,matmul_kernel,detail,metal-test,metal',
  "",
  header,
  ',_result,2,2026-10-26T03:00:00.000001Z,0.25,duration_ms,hush-test,hush.encoder',
  ',_result,2,2026-10-26T03:00:00.000001Z,128,bytes,hush-test,hush.encoder',
  ',_result,2,2026-10-26T03:00:00.000001Z,1,sequence,hush-test,hush.encoder'
].join("\r\n");
const events = parseCsv(csv);
assert.equal(events.length, 2);
assert.equal(events[0].detail, 'node=GRU shape=[1,32] quote="value"');
assert.equal(events[0].duration_ms, 0.25);
assert.equal(events[0].bytes, 128);
assert.deepEqual(runs(events).map(run => run.kind).sort(), ["hush", "old"]);
assert.equal(parseCsv("\r\n").length, 0);
assert.equal(parseCsv(header + "\n,_result,0,garbage,x,event,run,stage").length, 0);
const batchEvents = [
  { _time: "2026-10-26T03:00:00.000001Z", run_id: "hush", stage: "hush.batch", event: "stage_start", detail: "backend=metal frames=2 frame_start=0", duration_ms: 0, bytes: 0, sequence: 1 },
  { _time: "2026-10-26T03:00:00.010001Z", run_id: "hush", stage: "hush.frontend", event: "stage_complete", detail: "device=cpu spectrum=[1,161,2]", duration_ms: 1.2, bytes: 1928, sequence: 2 },
  { _time: "2026-10-26T03:00:00.020001Z", run_id: "hush", stage: "hush.batch", event: "stage_complete", detail: "backend=metal frames=2 frame_start=0", duration_ms: 20, bytes: 1280, sequence: 3 },
  { _time: "2026-10-26T03:00:00.030001Z", run_id: "hush", stage: "hush.state", event: "state_retained", detail: "recurrent=5 padding=2 pool_used_bytes=100", duration_ms: 0, bytes: 16, sequence: 4 }
];
assert.equal(phaseFor(batchEvents[1]), "frontend");
assert.equal(batches(batchEvents).length, 1);
assert.equal(batches(batchEvents)[0].frames, 2);
assert.equal(batches(batchEvents)[0].duration, 20);
assert.equal(milestones(batchEvents, false).length >= 2, true);
assert.equal(stageStats(batchEvents, "frontend").duration, 1.2);
console.log("Influx CSV reconstruction, quoted shapes, and run isolation passed");
