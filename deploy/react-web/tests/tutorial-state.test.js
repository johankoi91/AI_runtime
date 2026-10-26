import test from "node:test";
import assert from "node:assert/strict";

import lessons from "../src/lib/tutorial-lessons.js";
import { getFrameCount } from "../src/tutorial/canvas-engine.js";
import { createTutorialState, tutorialReducer } from "../src/tutorial/tutorial-state.js";

test("tutorial state starts from a lesson hash and preserves visited lessons", () => {
  const tensorIndex = lessons.findIndex(lesson => lesson.id === "tensor");
  const state = createTutorialState("tensor", JSON.stringify(["sound"]));
  assert.equal(state.selected, tensorIndex);
  assert.deepEqual(state.visited.sort(), ["sound", "tensor"]);
});

test("tutorial reducer resets parameters and advances animation frames", () => {
  let state = createTutorialState("conv");
  const initialFrames = getFrameCount(lessons[state.selected], state.parameters);
  state = tutorialReducer(state, { type: "control", id: "stride", value: 2 });
  const changedFrames = getFrameCount(lessons[state.selected], state.parameters);
  assert.notEqual(changedFrames, initialFrames);
  state = tutorialReducer(state, { type: "step" });
  assert.equal(state.frame, 1);
  state = tutorialReducer(state, { type: "choose", index: 0 });
  assert.equal(state.frame, 0);
  assert.equal(state.playing, false);
});
