function signal(length, cycles, amplitude) {
  return Array.from({length:length}, function (_, index) {
    return amplitude * Math.sin(2 * Math.PI * cycles * index / length) +
      .22 * Math.cos(2 * Math.PI * 5 * index / length);
  });
}
function vorbis(index, length) {
  var sine = Math.sin(Math.PI * (index + .5) / length);
  return Math.sin(Math.PI / 2 * sine * sine);
}
function windowed(values) {
  return values.map(function (value, index) { return value * vorbis(index, values.length); });
}
function complexMultiply(left, right) {
  return [left[0] * right[0] - left[1] * right[1], left[0] * right[1] + left[1] * right[0]];
}
function complexSum(values) {
  return values.reduce(function (sum, value) { return [sum[0] + value[0], sum[1] + value[1]]; }, [0, 0]);
}
function dft(values, count) {
  var length = values.length;
  return Array.from({length:count || length}, function (_, bin) {
    var sum = [0, 0];
    values.forEach(function (value, index) {
      var angle = -2 * Math.PI * bin * index / length;
      sum[0] += value * Math.cos(angle) / length;
      sum[1] += value * Math.sin(angle) / length;
    });
    return sum;
  });
}
function inverse(spectrum) {
  var length = spectrum.length;
  return Array.from({length:length}, function (_, index) {
    return spectrum.reduce(function (sum, bin, frequency) {
      var angle = 2 * Math.PI * frequency * index / length;
      return sum + bin[0] * Math.cos(angle) - bin[1] * Math.sin(angle);
    }, 0);
  });
}
function magnitudes(spectrum) {
  return spectrum.map(function (value) { return Math.hypot(value[0], value[1]); });
}
function bandEnergy(spectrum, first, last) {
  var selected = spectrum.slice(first, last);
  return selected.reduce(function (sum, bin) { return sum + bin[0] * bin[0] + bin[1] * bin[1]; }, 0) / selected.length;
}
function ema(previous, current, alpha) { return alpha * previous + (1 - alpha) * current; }
function sigmoid(value) { return 1 / (1 + Math.exp(-value)); }
function gru(input, previous, mode) {
  var update = sigmoid(.8 * input + .4 * previous - .2);
  var reset = sigmoid(-.5 * input + .7 * previous + .1);
  var candidate = Math.tanh(1.1 * input + (mode ? reset * (.6 * previous + .2) : .6 * (reset * previous) + .2));
  return {update:update, reset:reset, candidate:candidate, output:update * previous + (1 - update) * candidate};
}
function matrixMultiply(left, right) {
  return left.map(function (row) {
    return right[0].map(function (_, column) {
      return row.reduce(function (sum, value, inner) { return sum + value * right[inner][column]; }, 0);
    });
  });
}
function conv1d(values, weights, stride, padding) {
  var size = Math.floor((values.length + 2 * padding - weights.length) / stride) + 1;
  return Array.from({length:size}, function (_, index) {
    return weights.reduce(function (sum, weight, tap) {
      return sum + (values[index * stride + tap - padding] || 0) * weight;
    }, 0);
  });
}
function deepFilter(history, coefficients) {
  return complexSum(history.map(function (value, index) { return complexMultiply(value, coefficients[index]); }));
}
function rms(values) {
  return Math.sqrt(values.reduce(function (sum, value) { return sum + value * value; }, 0) / values.length);
}
function snr(reference, actual) {
  var energy = reference.reduce(function (sum, value) { return sum + value * value; }, 0);
  var error = reference.reduce(function (sum, value, index) { return sum + Math.pow(value - actual[index], 2); }, 0);
  return error === 0 ? Infinity : 10 * Math.log10(energy / error);
}
function strides(shape) {
  var result = [], size = 1;
  for (var index = shape.length - 1; index >= 0; index--) { result.unshift(size); size *= shape[index]; }
  return result;
}
var api = {signal:signal, vorbis:vorbis, windowed:windowed, complexMultiply:complexMultiply,
  complexSum:complexSum, dft:dft, inverse:inverse, magnitudes:magnitudes, bandEnergy:bandEnergy,
  ema:ema, sigmoid:sigmoid, gru:gru, matrixMultiply:matrixMultiply, conv1d:conv1d,
  deepFilter:deepFilter, rms:rms, snr:snr, strides:strides};

export { signal, vorbis, windowed, complexMultiply, complexSum, dft, inverse, magnitudes, bandEnergy, ema, sigmoid, gru, matrixMultiply, conv1d, deepFilter, rms, snr, strides };
export default api;
