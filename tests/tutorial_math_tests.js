const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const math = require("../deploy/web/tutorial-math.js");
function close(actual, expected, tolerance = 1e-10) {
  assert.ok(Math.abs(actual - expected) < tolerance, actual + " != " + expected);
}
assert.deepEqual(math.complexMultiply([1, 1], [1, -1]), [2, 0]);
close(Math.hypot(...math.complexMultiply([3, 4], [.3, .4])), 2.5);
const dc = math.dft(Array(32).fill(2));
close(dc[0][0], 2);
dc.slice(1).forEach(bin => close(Math.hypot(...bin), 0));
const sine = Array.from({length:32}, (_, index) => Math.sin(2 * Math.PI * 3 * index / 32));
const spectrum = math.dft(sine);
close(spectrum[3][0], 0);
close(spectrum[3][1], -.5);
close(spectrum[29][1], .5);
math.inverse(spectrum).forEach((value, index) => close(value, sine[index]));
for (let index = 0; index < 160; index++)
  close(math.vorbis(index, 320) ** 2 + math.vorbis(index + 160, 320) ** 2, 1);
const original = Array.from({length:48}, (_, index) => .8 * Math.sin(index * .4));
const first = math.inverse(math.dft(math.windowed(original.slice(0, 32))));
const second = math.inverse(math.dft(math.windowed(original.slice(16, 48))));
for (let index = 16; index < 32; index++)
  close(first[index] * math.vorbis(index, 32) + second[index - 16] * math.vorbis(index - 16, 32), original[index]);
assert.deepEqual(math.matrixMultiply([[1,2,3],[4,5,6]], [[1,2],[3,4],[5,6]]), [[22,28],[49,64]]);
assert.deepEqual(math.conv1d([1,2,-1,3,0,2,1], [1,0,-1], 1, 0), [2,-1,-1,1,-1]);
assert.deepEqual(math.conv1d([1,2,-1,3,0,2,1], [1,0,-1], 2, 1), [-2,-1,1,2]);
assert.deepEqual(math.strides([1,2,32,64]), [4096,2048,64,1]);
close(math.ema(-60, -70, .8), -62);
close(math.sigmoid(0), .5);
assert.notEqual(math.gru(.5, .7, 0).candidate, math.gru(.5, .7, 1).candidate);
for (const mode of [0, 1]) {
  const result = math.gru(.5, .7, mode);
  assert.ok(result.output >= Math.min(.7, result.candidate) && result.output <= Math.max(.7, result.candidate));
}
assert.deepEqual(math.deepFilter([[1,1], [1,-1]], [[1,0], [1,0]]), [2,0]);
close(math.bandEnergy([[1,0],[0,2]], 0, 2), 2.5);
close(math.rms([1,-1,1,-1]), 1);
close(math.snr([1,1], [1.1,.9]), 20);
const scope = {window:{}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../deploy/web/tutorial-lessons.js"), "utf8"), scope);
const lessons = scope.window.HushTutorialLessons;
assert.equal(lessons.length, 18);
assert.equal(new Set(lessons.map(lesson => lesson.id)).size, 18);
for (const lesson of lessons) {
  assert.ok(lesson.frames > 0 && lesson.equations.length > 0);
  assert.ok(lesson.quiz.answer >= 0 && lesson.quiz.answer < lesson.quiz.choices.length);
  for (const source of lesson.sources) assert.ok(fs.existsSync(path.join(__dirname, "..", source)), source);
}
console.log("Tutorial: DFT/inverse, Vorbis overlap-add, complex math, MatMul, Conv, GRU, normalization, metrics and 18 lessons passed");
