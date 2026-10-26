import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";

import lessons from "../src/lib/tutorial-lessons.js";
import math from "../src/lib/tutorial-math.js";
import { getFrameCount, renderTutorialFrame } from "../src/tutorial/canvas-engine.js";

const require = createRequire(import.meta.url);
const legacyMath = require("../../web/tutorial-math.js");
const readLegacy = filename => readFileSync(new URL(`../../web/${filename}`, import.meta.url), "utf8");
const tutorialSource = readLegacy("tutorial.js");
const drawingSource = tutorialSource.slice(tutorialSource.indexOf("  function fixed"), tutorialSource.indexOf("  function draw(announce)"));
const legacyRender = vm.runInNewContext(`(function(context, lesson, frame, fraction, parameters, width, height, colors, math) {
  var lessons = [lesson], selected = 0;
  ${drawingSource}
  context.clearRect(0, 0, width, height);
  return drawings[lesson.kind]();
})`);
const colors = { text: "#ecf1ef", muted: "#a2b7b8", green: "#9be5bd", blue: "#89c5f0", purple: "#c1b2f4", amber: "#f2c581", red: "#f19c9c", line: "#2a3c46", panel: "#101f29", stage: "#0d1a23" };

function recorder() {
  const calls = [];
  const context = new Proxy({}, {
    get: (_target, property) => (...argumentsList) => { calls.push([property, ...argumentsList]); },
    set: (_target, property, value) => { calls.push([property, value]); return true; }
  });
  return { calls, context };
}

test("both independent page styles remain byte-identical to the original", () => {
  for (const page of ["inference", "tutorial"]) {
    assert.equal(readFileSync(new URL(`../src/styles/${page}.css`, import.meta.url), "utf8"), readLegacy(`${page}.css`));
  }
});

test("all 18 lesson definitions and formulas preserve the original data", () => {
  const sandbox = { window: {} };
  vm.runInNewContext(readLegacy("tutorial-lessons.js"), sandbox);
  assert.equal(JSON.stringify(lessons), JSON.stringify(sandbox.window.HushTutorialLessons));
});

test("all lesson Canvas frames preserve drawing commands and calculation text", () => {
  let framesChecked = 0;
  for (const lesson of lessons) {
    const initial = Object.fromEntries(lesson.controls.map(control => [control.id, control.value]));
    const variants = [initial];
    for (const control of lesson.controls) {
      const values = control.options ? control.options.map(option => option[0]) : [control.min, control.max];
      for (const value of values) variants.push({ ...initial, [control.id]: value });
    }
    for (const parameters of variants) {
      const count = getFrameCount(lesson, parameters);
      for (const [width, height] of [[1134, 360], [358, 405]]) {
        for (let frame = 0; frame < count; frame++) {
          const fraction = frame % 2 ? 0.375 : 0;
          const actual = recorder();
          const expected = recorder();
          const result = renderTutorialFrame(actual.context, { lesson, frame, fraction, parameters, width, height, colors });
          const legacyResult = legacyRender(expected.context, lesson, frame, fraction, parameters, width, height, colors, legacyMath);
          assert.equal(result, legacyResult, `${lesson.id} frame ${frame}`);
          assert.deepEqual(actual.calls, expected.calls, `${lesson.id} frame ${frame} width ${width}`);
          framesChecked++;
        }
      }
    }
  }
  assert.ok(framesChecked >= lessons.length * 2);
  console.log(`Compared ${framesChecked} Canvas frames including control boundaries at desktop/mobile sizes`);
});

test("ported tutorial math preserves original fixed numerical results", () => {
  assert.deepEqual(math.dft(math.signal(32, 3, 0.7)), legacyMath.dft(legacyMath.signal(32, 3, 0.7)));
  assert.deepEqual(math.gru(0.5, 0.7, 1), legacyMath.gru(0.5, 0.7, 1));
});
