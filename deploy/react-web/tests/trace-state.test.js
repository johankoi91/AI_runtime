import test from "node:test";
import assert from "node:assert/strict";

import data from "../src/lib/trace-data.js";
import { csvFor, fixtureEvents } from "./fixtures.js";
import { createTraceState, traceReducer } from "../src/inference/trace-state.js";

test("fixture Trace data reconstructs Hush runs and two complete batches", () => {
  const events = data.parseCsv(csvFor(fixtureEvents()));
  const runs = data.runs(events);
  const hush = runs.find(run => run.kind === "hush");
  assert.ok(hush);
  const batches = data.batches(hush.events);
  assert.equal(batches.length, 2);
  assert.equal(batches[0].frames, 32);
  assert.equal(batches[0].complete, true);
  assert.equal(data.stageStats(batches[0].events, "encoder").devices.includes("metal"), true);
});

test("Trace reducer keeps the selected batch while toggling detail and playback", () => {
  const events = data.parseCsv(csvFor(fixtureEvents()));
  const runs = data.runs(events);
  let state = createTraceState();
  state = traceReducer(state, { type: "loaded", runs, manual: false });
  assert.equal(state.mode, "real");
  assert.equal(state.batches.length, 2);
  const selectedBatch = state.batch.id;
  state = traceReducer(state, { type: "details", value: true });
  assert.equal(state.batch.id, selectedBatch);
  assert.equal(state.events.some(event => event.event === "node_complete"), true);
  state = traceReducer(state, { type: "play" });
  assert.equal(state.playing, true);
  state = traceReducer(state, { type: "tick" });
  assert.equal(state.cursor, 1);
  state = traceReducer(state, { type: "lesson" });
  assert.equal(state.mode, "lesson");
});
