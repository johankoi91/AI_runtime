export const teachingSamples = [1, -1, 1, -1];

export function rmsCalculation(samples) {
  if (!samples.length || samples.some(value => !Number.isFinite(value))) {
    throw new RangeError("RMS requires a nonempty finite sample sequence");
  }
  const squares = samples.map(value => value * value);
  const sum = samples.reduce((total, value) => total + value, 0);
  const squareSum = squares.reduce((total, value) => total + value, 0);
  const meanSquare = squareSum / samples.length;
  return { samples, squares, sum, mean: sum / samples.length, squareSum, meanSquare, rms: Math.sqrt(meanSquare) };
}

export function mixingCalculation(sirDb, targetRms = 1, sourceRms = 1) {
  if (!Number.isFinite(sirDb) || !Number.isFinite(targetRms) || !Number.isFinite(sourceRms) || targetRms <= 0 || sourceRms <= 0) {
    throw new RangeError("SIR mixing requires finite dB and positive source RMS levels");
  }
  const powerRatio = 10 ** (sirDb / 10);
  const amplitudeRatio = 10 ** (-sirDb / 20);
  const gain = targetRms / sourceRms * amplitudeRatio;
  const backgroundRms = gain * sourceRms;
  const targetPower = targetRms ** 2;
  const backgroundPower = backgroundRms ** 2;
  const verifiedDb = 10 * Math.log10(targetPower / backgroundPower);
  return { sirDb, targetRms, sourceRms, powerRatio, amplitudeRatio, gain, backgroundRms, targetPower, backgroundPower, verifiedDb };
}
