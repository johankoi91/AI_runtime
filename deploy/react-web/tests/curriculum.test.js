import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import lessons, { modules } from "../src/lib/curriculum.js";
import originalLessons from "../src/lib/tutorial-lessons.js";
import walkthroughs from "../src/lib/formula-walkthroughs.js";
import { createTutorialState, tutorialReducer } from "../src/tutorial/tutorial-state.js";

test("seven modules teach dependencies before use and preserve existing lesson URLs", () => {
  assert.equal(modules.length, 7);
  assert.equal(lessons.length, 26);
  const positions = new Map(lessons.map((lesson, index) => [lesson.id, index]));
  assert.equal(positions.size, lessons.length);
  for (const [index, lesson] of lessons.entries()) {
    for (const prerequisite of lesson.prerequisites) assert.ok(positions.get(prerequisite) < index, `${prerequisite} must precede ${lesson.id}`);
    for (const source of lesson.sources) assert.ok(existsSync(new URL(`../../../${source}`, import.meta.url)), source);
  }
  for (const original of originalLessons) assert.ok(positions.has(original.id), original.id);
  assert.ok(positions.get("rms") < positions.get("decibels"));
  assert.ok(positions.get("vector") < positions.get("matmul"));
  assert.ok(positions.get("sine") < positions.get("dft"));
});

test("each lesson includes symbols, purpose, numerical steps and a valid checkpoint", () => {
  assert.deepEqual(new Set(Object.keys(walkthroughs)), new Set(lessons.map(lesson => lesson.id)));
  for (const lesson of lessons) {
    const formula = walkthroughs[lesson.id];
    for (const field of ["expression", "read", "why"]) assert.ok(formula[field].length > 5, `${lesson.id} ${field}`);
    assert.ok(formula.example.length >= 3, lesson.id);
    assert.ok(formula.variables.length >= 3, lesson.id);
    formula.variables.forEach(([symbol, meaning]) => assert.ok(symbol && meaning, lesson.id));
    assert.ok(lesson.quiz.choices[lesson.quiz.answer], lesson.id);
    assert.ok(lesson.quiz.reason && lesson.implementation && lesson.takeaway, lesson.id);
  }
});

test("new learners start at proportions and legacy anchors still select the same topic", () => {
  assert.equal(lessons[createTutorialState().selected].id, "ratio");
  assert.equal(lessons[createTutorialState("sound").selected].id, "sound");
  assert.equal(lessons[createTutorialState("missing").selected].id, "ratio");
  assert.deepEqual(createTutorialState("rms", "broken", '{"bad":true}').completed, []);
});

test("viewing and wrong answers never count as completion; correct answers persist independently", () => {
  let state = createTutorialState("ratio");
  assert.deepEqual(state.completed, []);
  assert.deepEqual(tutorialReducer(state, { type: "complete" }).completed, []);
  state = tutorialReducer(state, { type: "answer", index: 1 });
  assert.deepEqual(tutorialReducer(state, { type: "complete" }).completed, []);
  state = tutorialReducer(state, { type: "answer", index: 0 });
  state = tutorialReducer(state, { type: "complete" });
  assert.deepEqual(state.completed, ["ratio"]);
  assert.deepEqual(tutorialReducer(state, { type: "complete" }).completed, ["ratio"]);
  const restored = createTutorialState("rms", JSON.stringify(state.visited), JSON.stringify([...state.completed, "missing", "ratio"]));
  assert.deepEqual(restored.completed, ["ratio"]);
});

test("changing learning steps pauses playback; choosing a lesson resets its workspace", () => {
  let state = createTutorialState("sine");
  state = tutorialReducer(state, { type: "play" });
  state = tutorialReducer(state, { type: "learning-step", index: 2 });
  assert.equal(state.playing, false);
  assert.equal(state.learningStep, 2);
  state = tutorialReducer(state, { type: "choose", index: 0 });
  assert.equal(state.learningStep, 0);
  assert.equal(state.frame, 0);
  assert.equal(state.answer, null);
});
