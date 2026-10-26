const assert = require("node:assert/strict");
const { parseCsv, runs } = require("../deploy/web/trace_data.js");
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
console.log("Influx CSV reconstruction, quoted shapes, and run isolation passed");
