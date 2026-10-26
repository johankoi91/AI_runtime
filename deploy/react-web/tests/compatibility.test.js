import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";

import lessons from "../src/lib/tutorial-lessons.js";
import guidance from "../src/lib/lesson-guidance.js";
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

test("Trace style stays byte-identical and tutorial style keeps the original visual system", () => {
  assert.equal(readFileSync(new URL("../src/styles/inference.css", import.meta.url), "utf8"), readLegacy("inference.css"));
  const tutorialCss = readFileSync(new URL("../src/styles/tutorial.css", import.meta.url), "utf8");
  for (const selector of [".course {", ".stage {", "#animation {", ".reading-grid {", ".quiz {", "@media (max-width: 600px)"]) {
    assert.ok(tutorialCss.includes(selector), selector);
  }
  assert.ok(tutorialCss.includes(".formula-guide {"));
});

test("all 18 lesson definitions and formulas preserve the original data", () => {
  const sandbox = { window: {} };
  vm.runInNewContext(readLegacy("tutorial-lessons.js"), sandbox);
  assert.equal(JSON.stringify(lessons), JSON.stringify(sandbox.window.HushTutorialLessons));
});

test("every lesson has a complete formula guide for the detailed React tutorial", () => {
  assert.equal(Object.keys(guidance).length, lessons.length);
  for (const lesson of lessons) {
    const guide = guidance[lesson.id];
    assert.ok(guide.hook && guide.read && guide.example && guide.animation, lesson.id);
    assert.equal(guide.parts.length, 4, lesson.id);
    guide.parts.forEach(([symbol, explanation]) => {
      assert.ok(symbol && explanation, lesson.id);
    });
  }
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
