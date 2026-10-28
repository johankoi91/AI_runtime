import test from "node:test";
import assert from "node:assert/strict";
import { mixingCalculation, rmsCalculation, teachingSamples } from "../src/tutorial/sound-derivation-math.js";

test("a zero ordinary mean does not imply a zero RMS", () => {
  const result = rmsCalculation(teachingSamples);
  assert.equal(result.mean, 0);
  assert.deepEqual(result.squares, [1, 1, 1, 1]);
  assert.equal(result.squareSum, 4);
  assert.equal(result.meanSquare, 1);
  assert.equal(result.rms, 1);
});

test("doubling amplitude quadruples mean-square and doubles RMS", () => {
  const original = rmsCalculation(teachingSamples);
  const scaled = rmsCalculation(teachingSamples.map(value => value * 2));
  assert.equal(scaled.meanSquare, original.meanSquare * 4);
  assert.equal(scaled.rms, original.rms * 2);
});

test("uneven sample areas redistribute without changing total mean-square strength", () => {
  const result = rmsCalculation([1, -3, 3, -1]);
  assert.equal(result.mean, 0);
  assert.deepEqual(result.squares, [1, 9, 9, 1]);
  assert.equal(result.squareSum, 20);
  assert.equal(result.meanSquare, 5);
  assert.ok(Math.abs(result.rms - 2.23606797749979) < 1e-14);
  assert.ok(Math.abs(result.rms ** 2 * result.samples.length - result.squareSum) < 1e-13);
});

test("repeating the same samples changes total energy indicator but not RMS", () => {
  const original = rmsCalculation(teachingSamples);
  const repeated = rmsCalculation(teachingSamples.concat(teachingSamples));
  assert.equal(repeated.squareSum, original.squareSum * 2);
  assert.equal(repeated.meanSquare, original.meanSquare);
  assert.equal(repeated.rms, original.rms);
});

test("12 dB gives about 25.1 percent amplitude and 6.31 percent power", () => {
  const result = mixingCalculation(12);
  assert.ok(Math.abs(result.gain - 0.251188643150958) < 1e-14);
  assert.ok(Math.abs(result.backgroundPower - 0.0630957344480193) < 1e-14);
  assert.ok(Math.abs(result.verifiedDb - 12) < 1e-12);
  assert.ok(Math.abs(Math.sqrt(1 / result.powerRatio) - result.backgroundRms) < 1e-14);
});

test("unequal source levels require a gain correction and still meet requested SIR", () => {
  const equal = mixingCalculation(12, 1, 1);
  const unequal = mixingCalculation(12, 1, 2);
  assert.equal(unequal.gain, equal.gain / 2);
  assert.equal(unequal.backgroundRms, equal.backgroundRms);
  assert.ok(Math.abs(unequal.verifiedDb - 12) < 1e-12);
});

test("actual Hush demo target RMS 0.04 preserves SIR for unequal raw sources", () => {
  const result = mixingCalculation(18, 0.04, 0.12);
  assert.ok(Math.abs(result.backgroundRms - 0.04 * 10 ** (-18 / 20)) < 1e-14);
  assert.ok(Math.abs(result.verifiedDb - 18) < 1e-12);
});

test("larger SIR allows monotonically smaller background", () => {
  const values = [0, 6, 12, 18, 24].map(value => mixingCalculation(value));
  assert.equal(values[0].gain, 1);
  values.slice(1).forEach((value, index) => assert.ok(value.gain < values[index].gain));
  const halfAmplitudeDb = 20 * Math.log10(2);
  assert.ok(Math.abs(mixingCalculation(halfAmplitudeDb).gain - 0.5) < 1e-14);
});

test("empty RMS and silent source ratios are rejected rather than producing NaN", () => {
  assert.throws(() => rmsCalculation([]), RangeError);
  assert.throws(() => rmsCalculation([NaN]), RangeError);
  assert.throws(() => mixingCalculation(12, 1, 0), RangeError);
  assert.throws(() => mixingCalculation(Infinity), RangeError);
});
