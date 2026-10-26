(function () {
  "use strict";
  var lessons = window.HushTutorialLessons;
  var math = window.HushTutorialMath;
  var canvas = document.getElementById("animation");
  var context = canvas.getContext("2d");
  var styles = getComputedStyle(document.documentElement);
  var colors = {};
  ["text", "muted", "green", "blue", "purple", "amber", "red", "line", "panel", "stage"].forEach(function (key) {
    colors[key] = styles.getPropertyValue("--" + key).trim();
  });
  var selected = Math.max(0, lessons.findIndex(function (lesson) { return lesson.id === location.hash.slice(1); }));
  var frame = 0, playing = false, lastTick = 0, fraction = 0, parameters = {}, visited = new Set();
  var width = 800, height = 360;
  var $ = function (id) { return document.getElementById(id); };
  function escape(value) {
    return String(value).replace(/[&<>"']/g, function (character) {
      return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character];
    });
  }
  function fixed(value, digits) { return Number(value).toFixed(digits === undefined ? 3 : digits); }
  function complex(value) {
    return fixed(value[0], 2) + (value[1] < 0 ? " − " : " + ") + fixed(Math.abs(value[1]), 2) + "i";
  }
  function text(value, x, y, color, align, size) {
    context.font = (size || 12) + 'px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    context.fillStyle = color || colors.text;
    context.textAlign = align || "left";
    context.textBaseline = "middle";
    context.fillText(String(value), x, y);
  }
  function line(firstX, firstY, secondX, secondY, color, thickness) {
    context.beginPath(); context.moveTo(firstX, firstY); context.lineTo(secondX, secondY);
    context.strokeStyle = color || colors.line; context.lineWidth = thickness || 1; context.stroke();
  }
  function rect(x, y, boxWidth, boxHeight, fill, stroke) {
    context.fillStyle = fill || colors.panel; context.fillRect(x, y, boxWidth, boxHeight);
    if (stroke) { context.strokeStyle = stroke; context.lineWidth = 1; context.strokeRect(x, y, boxWidth, boxHeight); }
  }
  function point(x, y, radius, color) {
    context.beginPath(); context.arc(x, y, radius, 0, Math.PI * 2);
    context.fillStyle = color || colors.green; context.fill();
  }
  function arrow(firstX, firstY, secondX, secondY, color) {
    line(firstX, firstY, secondX, secondY, color, 1.5);
    var angle = Math.atan2(secondY - firstY, secondX - firstX);
    line(secondX, secondY, secondX - 7 * Math.cos(angle - .5), secondY - 7 * Math.sin(angle - .5), color, 1.5);
    line(secondX, secondY, secondX - 7 * Math.cos(angle + .5), secondY - 7 * Math.sin(angle + .5), color, 1.5);
  }
  function chart(x, y, chartWidth, chartHeight, minimum, maximum, caption) {
    line(x, y + chartHeight, x + chartWidth, y + chartHeight);
    line(x, y, x, y + chartHeight);
    if (minimum < 0 && maximum > 0) line(x, y + chartHeight * maximum / (maximum - minimum), x + chartWidth, y + chartHeight * maximum / (maximum - minimum));
    text(caption, x, y - 14, colors.muted, "left", 11);
    text(fixed(maximum, 1), x - 5, y + 3, colors.muted, "right", 11);
    text(fixed(minimum, 1), x - 5, y + chartHeight - 3, colors.muted, "right", 11);
    return {
      x: function (index, count) { return x + index / Math.max(1, count - 1) * chartWidth; },
      y: function (value) { return y + chartHeight - (value - minimum) / (maximum - minimum) * chartHeight; },
      width:chartWidth, height:chartHeight
    };
  }
  function plot(values, axes, color, count) {
    context.beginPath();
    values.slice(0, count === undefined ? values.length : count).forEach(function (value, index) {
      if (index === 0) context.moveTo(axes.x(index, values.length), axes.y(value));
      else context.lineTo(axes.x(index, values.length), axes.y(value));
    });
    context.strokeStyle = color || colors.green; context.lineWidth = 2; context.stroke();
  }
  function bar(x, y, barWidth, barHeight, ratio, color) {
    rect(x, y, barWidth, barHeight, colors.line);
    rect(x, y, barWidth * Math.max(0, Math.min(1, ratio)), barHeight, color || colors.green);
  }
  function sampleSignal() { return math.signal(32, 3, .7); }
  function frames() {
    if (lessons[selected].kind === "conv")
      return math.conv1d([1,2,-1,3,0,2,1], [1,0,-1], Number(parameters.stride), Number(parameters.padding)).length;
    return lessons[selected].frames;
  }
  function drawSound() {
    var attenuation = Math.pow(10, -Number(parameters.sir) / 20);
    var main = Array.from({length:160}, function (_, index) { return .65 * Math.sin(2 * Math.PI * 3 * index / 160); });
    var mixed = main.map(function (value, index) { return value + attenuation * .65 * Math.sin(2 * Math.PI * 11 * index / 160); });
    var axes = chart(38, 52, width - 65, height - 120, -1.4, 1.4, "振幅：蓝=主讲教学波，绿=混合波");
    plot(main, axes, colors.blue); plot(mixed, axes, colors.green);
    var limit = Math.min(159, Math.floor((frame + fraction) / 12 * 160));
    for (var index = 0; index <= limit; index += 4) point(axes.x(index, 160), axes.y(mixed[index]), 3, colors.amber);
    text("0", 38, height - 45, colors.muted); text("160 samples / 10 ms", width - 27, height - 45, colors.muted, "right");
    return "背景幅度比 a=10^(−" + parameters.sir + "/20)=" + fixed(attenuation) + "；160 个点仍是同一条混合音频。";
  }
  function drawTensor() {
    var columns = parameters.layout === "32" ? 2 : 3, rows = 6 / columns;
    var cell = Math.min(62, (width - 70) / 6), left = 30, top = 55;
    text("shape=[" + rows + "," + columns + "] · stride=[" + columns + ",1]", 24, 27, colors.blue);
    for (var index = 0; index < 6; index++) {
      var x = left + index % columns * (cell + 6), y = top + Math.floor(index / columns) * (cell + 6);
      rect(x, y, cell, cell, index === frame ? "#314637" : colors.panel, index === frame ? colors.green : colors.line);
      text(index + 1, x + cell / 2, y + cell / 2, colors.text, "center", 17);
    }
    var memoryY = height - 89, memoryCell = (width - 60) / 6;
    text("底层连续 Buffer：数字顺序没变", 24, memoryY - 20, colors.muted);
    for (var memoryIndex = 0; memoryIndex < 6; memoryIndex++) {
      rect(24 + memoryIndex * memoryCell, memoryY, memoryCell - 3, 36, memoryIndex === frame ? "#314637" : colors.panel, colors.line);
      text(memoryIndex + 1, 24 + memoryIndex * memoryCell + (memoryCell - 3) / 2, memoryY + 18);
    }
    return "index=(" + Math.floor(frame / columns) + "," + frame % columns + ") → offset=" +
      Math.floor(frame / columns) + "×" + columns + "+" + frame % columns + "=" + frame + "；字节偏移=" + frame * 4 + " B。";
  }
  function drawComplex() {
    var centerX = width / 2, centerY = height / 2 + 8, unit = Math.min(width * .21, height * .29);
    var angle = Number(parameters.angle) * Math.PI / 180 * Math.min(1, (frame + fraction + 1) / 12);
    var multiplier = [Number(parameters.gain) * Math.cos(angle), Number(parameters.gain) * Math.sin(angle)];
    var input = [1, .5], output = math.complexMultiply(input, multiplier);
    line(centerX - unit * 1.9, centerY, centerX + unit * 1.9, centerY);
    line(centerX, centerY - unit * 1.55, centerX, centerY + unit * 1.45);
    context.beginPath(); context.arc(centerX, centerY, unit, 0, Math.PI * 2); context.strokeStyle = colors.line; context.stroke();
    arrow(centerX, centerY, centerX + unit * input[0], centerY - unit * input[1], colors.blue);
    arrow(centerX, centerY, centerX + unit * output[0], centerY - unit * output[1], colors.green);
    line(centerX + unit * output[0], centerY, centerX + unit * output[0], centerY - unit * output[1], colors.amber);
    text("real", centerX + unit * 1.9, centerY + 19, colors.muted, "right");
    text("imag", centerX + 9, centerY - unit * 1.55, colors.muted);
    text("蓝：1+0.5i · 绿：旋转并缩放后的结果", 22, 27, colors.muted);
    return "(" + complex(input) + ") × (" + complex(multiplier) + ") = " + complex(output) + "。";
  }
  function drawWindow() {
    var values = math.signal(320, 3, .65), windows = values.map(function (_, index) { return math.vorbis(index, 320); });
    var result = math.windowed(values);
    var axes = chart(38, 52, width - 65, height - 120, -1.1, 1.1, "蓝=原始片段，黄=窗系数，绿=乘窗结果");
    plot(values, axes, colors.blue); plot(windows, axes, colors.amber); plot(result, axes, colors.green);
    var cursor = Math.min(319, Math.floor((frame + fraction + 1) / 16 * 320) - 1);
    line(axes.x(cursor, 320), axes.y(-1.1), axes.x(cursor, 320), axes.y(1.1), colors.text);
    point(axes.x(cursor, 320), axes.y(result[cursor]), 4, colors.green);
    text("历史 160 点", 38, height - 42, colors.muted);
    text("当前 160 点", width - 27, height - 42, colors.muted, "right");
    return "n=" + cursor + "：" + fixed(values[cursor]) + " × w[" + cursor + "]=" + fixed(windows[cursor]) +
      " → " + fixed(result[cursor]) + "。分析 20 ms，每次前进 10 ms。";
  }
  function drawDft() {
    var values = sampleSignal(), bin = Number(parameters.bin), spectrum = math.dft(values, 17);
    var reference = values.map(function (_, index) { return Math.cos(2 * Math.PI * bin * index / 32); });
    var firstAxes = chart(38, 50, width - 65, height * .33, -1.15, 1.15, "教学 N=32：蓝=输入，黄=当前 bin 的余弦参考");
    plot(values, firstAxes, colors.blue); plot(reference, firstAxes, colors.amber);
    var count = Math.min(32, (frame + 1) * 2), real = 0, imaginary = 0;
    for (var index = 0; index < count; index++) {
      real += values[index] * Math.cos(2 * Math.PI * bin * index / 32) / 32;
      imaginary -= values[index] * Math.sin(2 * Math.PI * bin * index / 32) / 32;
    }
    line(firstAxes.x(count - 1, 32), firstAxes.y(-1.15), firstAxes.x(count - 1, 32), firstAxes.y(1.15), colors.green);
    var magnitude = math.magnitudes(spectrum), bottom = height - 44, chartTop = height * .64;
    text("全窗幅度谱（参考答案），选中 bin " + bin, 38, chartTop - 19, colors.muted, "left", 11);
    var cell = (width - 65) / 17;
    magnitude.forEach(function (value, index) {
      var barHeight = value / .5 * (bottom - chartTop);
      rect(38 + index * cell, bottom - barHeight, Math.max(2, cell - 3), Math.max(1, barHeight), index === bin ? colors.amber : colors.green);
      if (index % 2 === 0) text(index, 38 + index * cell + cell / 2, bottom + 14, colors.muted, "center", 11);
    });
    return "前 " + count + "/32 项的累加：Re=" + fixed(real) + "，Im=" + fixed(imaginary) +
      "；全窗 bin " + bin + " 的幅度=" + fixed(magnitude[bin]) + "。";
  }
  function drawErb() {
    var spectrum = math.dft(sampleSignal(), 17), powers = spectrum.map(function (bin) { return bin[0] * bin[0] + bin[1] * bin[1]; });
    var ranges = [[0,4],[4,8],[8,12],[12,17]], active = Number(parameters.band);
    var bottom = height * .49, cell = (width - 60) / 17, maximum = Math.max.apply(null, powers);
    text("教学频点功率 |X[k]|² · 黄=选中 band", 24, 28, colors.muted);
    powers.forEach(function (value, index) {
      var barHeight = value / maximum * (height * .29);
      rect(30 + index * cell, bottom - barHeight, Math.max(2, cell - 3), Math.max(1, barHeight),
        index >= ranges[active][0] && index < ranges[active][1] ? colors.amber : colors.blue);
      if (index % 4 === 0) text(index, 30 + index * cell, bottom + 15, colors.muted, "left", 11);
    });
    var bandY = height * .65, bandWidth = (width - 56) / 4;
    ranges.forEach(function (range, index) {
      var energy = math.bandEnergy(spectrum, range[0], range[1]);
      var db = 10 * Math.log10(energy + 1e-10);
      rect(24 + index * bandWidth, bandY, bandWidth - 6, 65, colors.panel, index === active ? colors.amber : colors.line);
      text("band " + index, 30 + index * bandWidth, bandY + 18, index === active ? colors.amber : colors.muted, "left", 11);
      text(fixed(db, 1) + " dB", 30 + index * bandWidth, bandY + 44, colors.text, "left", 12);
    });
    var range = ranges[active], energy = math.bandEnergy(spectrum, range[0], range[1]);
    return "band " + active + "：bin " + range[0] + "…" + (range[1] - 1) + " 功率均值=" +
      fixed(energy, 6) + "；10log10(E+10⁻¹⁰)=" + fixed(10 * Math.log10(energy + 1e-10), 2) + " dB。";
  }
  function drawNorm() {
    var values = Array.from({length:16}, function (_, index) { return index < 8 ? -70 : -35; });
    var mean = -60, means = [];
    values.forEach(function (value) { mean = math.ema(mean, value, Number(parameters.alpha)); means.push(mean); });
    var axes = chart(38, 54, width - 65, height - 125, -90, -20, "教学 ERB dB：蓝=输入，绿=运行均值");
    plot(values, axes, colors.blue); plot(means, axes, colors.green, frame + 1);
    point(axes.x(frame, 16), axes.y(means[frame]), 5, colors.amber);
    text("时间步 0", 38, height - 40, colors.muted); text("15", width - 27, height - 40, colors.muted, "right");
    return "α=" + parameters.alpha + "；第 " + frame + " 帧 L=" + values[frame] + " dB，μ=" +
      fixed(means[frame], 2) + "；ERB 特征 (L−μ)/40=" + fixed((values[frame] - means[frame]) / 40) + "。";
  }
  function drawMatmul() {
    var left = [[1,2,3],[4,5,6]], right = [[1,2],[3,4],[5,6]], output = math.matrixMultiply(left, right);
    var element = Math.floor(frame / 3), row = Math.floor(element / 2), column = element % 2, inner = frame % 3;
    var cell = Math.min(47, (width - 80) / 7), top = 94, leftX = 18, rightX = leftX + cell * 3 + 20, outputX = rightX + cell * 2 + 20;
    text("A [2,3]", leftX, 49, colors.blue); text("B [3,2]", rightX, 49, colors.purple); text("Y [2,2]", outputX, 49, colors.green);
    [left, right, output].forEach(function (matrix, matrixIndex) {
      var origin = [leftX, rightX, outputX][matrixIndex];
      matrix.forEach(function (values, matrixRow) {
        values.forEach(function (value, matrixColumn) {
          var active = matrixIndex === 0 ? matrixRow === row && matrixColumn === inner :
            matrixIndex === 1 ? matrixRow === inner && matrixColumn === column : matrixRow === row && matrixColumn === column;
          rect(origin + matrixColumn * cell, top + matrixRow * cell, cell - 4, cell - 4, active ? "#394735" : colors.panel, active ? colors.amber : colors.line);
          var displayed = value;
          if (matrixIndex === 2) {
            var linear = matrixRow * 2 + matrixColumn;
            displayed = linear > element ? "·" : linear === element ?
              left[row].slice(0, inner + 1).reduce(function (sum, item, index) { return sum + item * right[index][column]; }, 0) : value;
          }
          text(displayed, origin + matrixColumn * cell + (cell - 4) / 2, top + matrixRow * cell + (cell - 4) / 2,
            active ? colors.amber : colors.text, "center", 14);
        });
      });
    });
    text("高亮元素相乘，然后加入当前输出格子。", 20, Math.min(height - 44, top + 3 * cell + 34), colors.muted);
    var terms = left[row].slice(0, inner + 1).map(function (value, index) { return value + "×" + right[index][column]; });
    var partial = left[row].slice(0, inner + 1).reduce(function (sum, value, index) { return sum + value * right[index][column]; }, 0);
    return "Y[" + row + "," + column + "] 累加：" + terms.join(" + ") + " = " + partial +
      "；完成这个格子需要 3 项。";
  }
  function drawConv() {
    var input = [1,2,-1,3,0,2,1], weights = [1,0,-1];
    var stride = Number(parameters.stride), padding = Number(parameters.padding);
    var output = math.conv1d(input, weights, stride, padding);
    var count = input.length + padding * 2, cell = Math.min(52, (width - 40) / count), left = (width - count * cell) / 2;
    text("一维教学卷积 · 蓝=输入，黄=当前 kernel 覆盖", 20, 28, colors.muted);
    for (var index = 0; index < count; index++) {
      var covered = index >= frame * stride && index < frame * stride + 3;
      rect(left + index * cell, 65, cell - 4, 42, covered ? "#453c29" : colors.panel, covered ? colors.amber : colors.line);
      text(input[index - padding] || 0, left + index * cell + (cell - 4) / 2, 86, covered ? colors.amber : colors.text, "center", 14);
      if (index < padding || index >= input.length + padding) text("pad", left + index * cell + cell / 2, 120, colors.muted, "center", 11);
    }
    weights.forEach(function (weight, index) {
      var x = left + (frame * stride + index) * cell;
      rect(x, 148, cell - 4, 35, "#2a273e", colors.purple);
      text("×" + weight, x + (cell - 4) / 2, 166, colors.purple, "center", 12);
    });
    var outputCell = Math.min(53, (width - 40) / output.length);
    text("输出", 20, 219, colors.green);
    output.forEach(function (value, index) {
      var x = 20 + index * outputCell;
      rect(x, 239, outputCell - 4, 40, index === frame ? "#314637" : colors.panel, colors.line);
      text(index <= frame ? value : "·", x + (outputCell - 4) / 2, 259, colors.green, "center", 14);
    });
    var values = weights.map(function (_, index) { return input[frame * stride + index - padding] || 0; });
    return "Y[" + frame + "]=" + values.map(function (value, index) { return "(" + value + "×" + weights[index] + ")"; }).join("+") +
      "=" + output[frame] + "；输出长度=" + output.length + "。";
  }
  function drawActivation() {
    var kind = parameters.activation;
    var calculate = kind === "relu" ? function (value) { return Math.max(0, value); } :
      kind === "sigmoid" ? math.sigmoid : Math.tanh;
    var minimum = kind === "relu" ? -.4 : kind === "sigmoid" ? -.15 : -1.2;
    var maximum = kind === "relu" ? 4.4 : kind === "sigmoid" ? 1.15 : 1.2;
    var values = Array.from({length:161}, function (_, index) { return calculate(-4 + index / 20); });
    var axes = chart(40, 48, width - 68, height - 112, minimum, maximum, "函数曲线 · 横轴输入 x，纵轴输出 y");
    plot(values, axes, colors.green);
    var input = -4 + frame / 15 * 8, output = calculate(input), x = axes.x(frame, 16), y = axes.y(output);
    line(x, axes.y(minimum), x, y, colors.amber);
    point(x, y, 6, colors.amber);
    text("−4", 40, height - 42, colors.muted); text("x=0", width / 2, height - 42, colors.muted, "center");
    text("+4", width - 28, height - 42, colors.muted, "right");
    return kind + "(" + fixed(input, 2) + ")=" + fixed(output, 4) + "；逐元素计算，shape 不变。";
  }
  function drawGru() {
    var previous = .7, rows = [];
    for (var index = 0; index <= frame; index++) {
      var result = math.gru(Number(parameters.input), previous, Number(parameters.lbr));
      rows.push({previous:previous, result:result}); previous = result.output;
    }
    var current = rows[frame], gates = [
      ["z 旧记忆比例", current.result.update, colors.amber],
      ["r 重置比例", current.result.reset, colors.purple],
      ["n 新候选", current.result.candidate, colors.green]
    ];
    text("标量教学 GRU · 权重固定，逐时间步递推", 20, 27, colors.muted);
    gates.forEach(function (gate, index) {
      var top = 64 + index * 49;
      text(gate[0], 20, top, gate[2], "left", 12);
      bar(130, top - 7, width - 205, 14, index === 2 ? (gate[1] + 1) / 2 : gate[1], gate[2]);
      text(fixed(gate[1]), width - 20, top, colors.text, "right", 12);
    });
    var axes = chart(40, height - 93, width - 68, 44, -1, 1, "绿色：hidden 历史");
    var history = rows.map(function (row) { return row.result.output; });
    while (history.length < 8) history.push(history[history.length - 1]);
    plot(history, axes, colors.green, rows.length);
    point(axes.x(frame, 8), axes.y(current.result.output), 4, colors.green);
    return "t=" + frame + "：h=" + fixed(current.result.update) + "×" + fixed(current.previous) +
      "+(1−" + fixed(current.result.update) + ")×" + fixed(current.result.candidate) + "=" + fixed(current.result.output) + "。";
  }
  function drawGraphs() {
    var compact = width < 620;
    var nodeWidth = compact ? 82 : 140;
    var positions = compact ? [[20,50],[width / 2 - 41,50],[20,170],[width - 102,170],[width / 2 - 41,280]] :
      [[20,115],[width * .25,115],[width * .52,45],[width * .52,210],[width - 166,115]];
    var labels = [["Frontend","ERB + DF"],["enc.onnx","共享特征"],["erb_dec","m / mask"],["df_dec","coefs"],["频域增强","Mask + DF"]];
    [[0,1],[1,2],[1,3],[2,4],[3,4]].forEach(function (pair) {
      var first = positions[pair[0]], second = positions[pair[1]];
      arrow(first[0] + nodeWidth / 2, first[1] + 50, second[0] + nodeWidth / 2, second[1], colors.line);
    });
    labels.forEach(function (labels, index) {
      var position = positions[index], active = index === Math.min(4, frame);
      rect(position[0], position[1], nodeWidth, 52, active ? "#314637" : colors.panel, active ? colors.green : colors.line);
      text(labels[0], position[0] + nodeWidth / 2, position[1] + 17, active ? colors.green : colors.text, "center", 12);
      text(labels[1], position[0] + nodeWidth / 2, position[1] + 37, colors.muted, "center", 11);
    });
    return "当前： " + labels[Math.min(4, frame)][0] + "。两张 decoder 共享 enc 输出；这里只画依赖，当前实现按 ERB → DF 顺序执行。";
  }
  function drawMask() {
    var magnitudes = math.magnitudes(math.dft(sampleSignal(), 17)), gain = Number(parameters.gain);
    var left = 30, cell = (width - 60) / 17, bottom = height - 75, span = height - 145, maximum = Math.max.apply(null, magnitudes);
    text("教学统一 gain：灰=原始幅度，绿=增强幅度", 20, 27, colors.muted);
    magnitudes.forEach(function (value, index) {
      var barHeight = value / maximum * span;
      rect(left + index * cell, bottom - barHeight, cell - 3, Math.max(1, barHeight), colors.line);
      rect(left + index * cell, bottom - barHeight * gain, cell - 3, Math.max(1, barHeight * gain), index === frame * 2 ? colors.amber : colors.green);
      if (index % 4 === 0) text(index, left + index * cell + cell / 2, bottom + 19, colors.muted, "center", 11);
    });
    text("频率 bin（教学 N=32）", width - 27, height - 25, colors.muted, "right", 11);
    return "(3+4i) × " + gain + " = " + complex([3 * gain, 4 * gain]) +
      "；幅度=" + fixed(5 * gain, 2) + "，功率=" + fixed(25 * gain * gain, 2) + "。此 gain 不是 Hush 的可调参数。";
  }
  function drawDeep() {
    var history = [[1,.2],[.8,-.3],[1.2,.1],[.6,.4],[1,-.2]];
    var coefficients = [[.1,0],[.15,.05],[.2,-.1],[.15,0],[.4,.1]];
    var products = history.map(function (value, index) { return math.complexMultiply(coefficients[index], value); });
    var cell = (width - 40) / 5;
    text("同一个低频 bin · 从最旧到当前", 20, 27, colors.muted);
    for (var index = 0; index < 5; index++) {
      var x = 20 + index * cell;
      rect(x, 63, cell - 5, 75, index === frame ? "#453c29" : colors.panel, index === frame ? colors.amber : colors.line);
      text(index === 4 ? "当前 t" : "t−" + (4 - index), x + cell / 2 - 3, 82, index === frame ? colors.amber : colors.muted, "center", 11);
      text(fixed(history[index][0], 1), x + cell / 2 - 3, 106, colors.blue, "center", 12);
      text((history[index][1] < 0 ? "" : "+") + fixed(history[index][1], 1) + "i", x + cell / 2 - 3, 124, colors.blue, "center", 11);
      text("× C" + index, x + cell / 2 - 3, 164, colors.purple, "center", 12);
      text(index <= frame ? fixed(products[index][0], 2) : "·", x + cell / 2 - 3, 207, colors.green, "center", 12);
      text(index <= frame ? (products[index][1] < 0 ? "" : "+") + fixed(products[index][1], 2) + "i" : "", x + cell / 2 - 3, 228, colors.green, "center", 11);
    }
    var accumulated = math.complexSum(products.slice(0, frame + 1));
    text("Σ 前 " + (frame + 1) + " 项 = " + complex(accumulated), 20, height - 47, colors.green, "left", 15);
    return "tap " + frame + "：" + complex(coefficients[frame]) + " × (" + complex(history[frame]) + ") = " +
      complex(products[frame]) + "；累计 " + complex(accumulated) + "。";
  }
  function drawIstft() {
    var length = 32, hop = 16;
    var original = Array.from({length:48}, function (_, index) { return .8 * Math.sin(index * .4); });
    var first = math.inverse(math.dft(math.windowed(original.slice(0, length)))).map(function (value, index) { return value * math.vorbis(index, length); });
    var second = math.inverse(math.dft(math.windowed(original.slice(hop, hop + length)))).map(function (value, index) { return value * math.vorbis(index, length); });
    var contributionA = original.map(function (_, index) { return index < length ? first[index] : 0; });
    var contributionB = original.map(function (_, index) { return index >= hop ? second[index - hop] : 0; });
    var combined = contributionA.map(function (value, index) { return value + contributionB[index]; });
    var axes = chart(40, 55, width - 67, height - 130, -1, 1, "直通教学 N=32：蓝/紫=两窗贡献，绿=相加");
    plot(original, axes, colors.line); plot(contributionA, axes, colors.blue); plot(contributionB, axes, colors.purple); plot(combined, axes, colors.green);
    var cursor = 16 + frame, currentX = axes.x(cursor, 48);
    line(currentX, axes.y(-1), currentX, axes.y(1), colors.amber);
    point(currentX, axes.y(combined[cursor]), 5, colors.amber);
    text("只在完整重叠区比较直通重建", 40, height - 42, colors.muted);
    return "重叠位置 n=" + cursor + "：" + fixed(contributionA[cursor]) + "+" + fixed(contributionB[cursor]) +
      "=" + fixed(combined[cursor]) + "；原始 " + fixed(original[cursor]) +
      "，误差 " + Math.abs(combined[cursor] - original[cursor]).toExponential(1) + "。";
  }
  function drawGpu() {
    var compact = width < 620, threadTop = compact ? 205 : 78;
    var threadLeft = compact ? 24 : width * .63, threadWidth = compact ? (width - 48) / 2 : width * .16;
    var pipeline = ["准备输入","复制到 Buffer","encoder 记录","commit / dispatch","第 1 轮乘加","第 2/3 轮乘加","wait + copy back"];
    text(pipeline[frame], 20, 28, colors.green, "left", 14);
    var bufferWidth = compact ? width - 48 : width * .5;
    [["input A","1  2  3  /  4  5  6"],["weights B","1  2  /  3  4  /  5  6"],["output Y",frame < 5 ? "·  ·  /  ·  ·" : "22 28 / 49 64"]].forEach(function (row, index) {
      var top = 58 + index * 43;
      rect(24, top, bufferWidth, 34, colors.panel, index === 2 ? colors.green : colors.line);
      text(row[0], 32, top + 10, colors.muted, "left", 11);
      text(row[1], 32, top + 25, index === 2 ? colors.green : colors.blue, "left", 11);
    });
    var left = [[1,2,3],[4,5,6]], right = [[1,2],[3,4],[5,6]];
    for (var index = 0; index < 4; index++) {
      var x = threadLeft + index % 2 * threadWidth, y = threadTop + Math.floor(index / 2) * 57;
      var active = frame >= 3;
      rect(x, y, threadWidth - 6, 49, active ? "#282f45" : colors.panel, active ? colors.purple : colors.line);
      text("thread(" + index % 2 + "," + Math.floor(index / 2) + ")", x + (threadWidth - 6) / 2, y + 15, colors.purple, "center", 11);
      var terms = frame < 4 ? 0 : frame === 4 ? 1 : 3;
      var partial = left[Math.floor(index / 2)].slice(0, terms).reduce(function (sum, value, inner) {
        return sum + value * right[inner][index % 2];
      }, 0);
      text(frame >= 4 ? (terms === 3 ? "Y=" : "sum=") + partial : active ? "已提交" : "未提交",
        x + (threadWidth - 6) / 2, y + 34, active ? colors.green : colors.muted, "center", 12);
    }
    if (!compact) text("四个逻辑输出位置；不等同四个物理核心", 24, height - 49, colors.muted);
    return "教学 grid=2×2。阶段 " + (frame + 1) + "/7：" + pipeline[frame] +
      "。每个线程内部累加 K=3；显示的是算法分工，不是硬件实测轨迹。";
  }
  function drawState() {
    var actions = ["第 0 帧写临时值","更新 GRU","独立复制历史","pool.reset()","第 1 帧覆盖临时区","读取独立历史","推进新的流状态"];
    var memoryWidth = width - 48;
    text(actions[frame], 20, 28, colors.green, "left", 14);
    text("临时 arena（会覆盖）", 24, 64, colors.muted);
    for (var index = 0; index < 3; index++) {
      var x = 24 + index * memoryWidth / 3;
      rect(x, 83, memoryWidth / 3 - 5, 52, "#293743", colors.line);
      text(frame === 3 ? "可复用" : frame > 3 ? "frame 1" : "frame 0", x + (memoryWidth / 3 - 5) / 2, 109, colors.blue, "center", 12);
    }
    text("独立状态区（跨帧保存）", 24, 167, colors.muted);
    ["GRU","causal Pad","DF","OLA"].forEach(function (label, index) {
      var boxWidth = (memoryWidth - 8) / 2, x = 24 + index % 2 * (boxWidth + 8), y = 188 + Math.floor(index / 2) * 57;
      rect(x, y, boxWidth, 45, frame >= 2 ? "#20392f" : colors.panel, frame >= 2 ? colors.green : colors.line);
      text(label, x + 10, y + 15, colors.green, "left", 12);
      text(frame >= 6 ? "state 1" : frame >= 2 ? "保留 state 0" : "初始化", x + 10, y + 33, colors.muted, "left", 11);
    });
    return "第 " + (frame + 1) + " 步：" + actions[frame] +
      "。地址的所有权不等于历史数值不被覆盖；状态必须另存。";
  }
  function drawQuality() {
    var duration = Number(parameters.compute), span = width - 135, maximum = 640;
    text("同一时间尺度（ms）· 纯教学输入", 20, 28, colors.muted);
    text("音频", 24, 80, colors.blue);
    bar(78, 69, span, 21, 320 / maximum, colors.blue);
    text("320 ms", width - 20, 80, colors.blue, "right", 12);
    text("计算", 24, 131, colors.green);
    bar(78, 120, span, 21, duration / maximum, colors.green);
    text(duration + " ms", width - 20, 131, colors.green, "right", 12);
    var reference = sampleSignal(), actual = reference.map(function (value, index) { return value + .001 * Math.sin(index * 1.7 + frame * .1); });
    var axes = chart(40, 206, width - 67, height - 253, -1, 1, "灰=参考，绿=有微小误差的教学结果");
    plot(reference, axes, colors.line); plot(actual, axes, colors.green);
    return "RTF=" + fixed(duration / 320, 2) + "，实时倍率=" + fixed(320 / duration, 2) +
      "×；批摊销=" + fixed(duration / 32, 2) + " ms/帧（不是单帧延迟）；教学误差 SNR=" + fixed(math.snr(reference, actual), 1) + " dB。";
  }
  var drawings = {sound:drawSound, tensor:drawTensor, complex:drawComplex, window:drawWindow, dft:drawDft,
    erb:drawErb, norm:drawNorm, matmul:drawMatmul, conv:drawConv, activation:drawActivation,
    gru:drawGru, graphs:drawGraphs, mask:drawMask, deep:drawDeep, istft:drawIstft,
    gpu:drawGpu, state:drawState, quality:drawQuality};
  function draw(announce) {
    context.clearRect(0, 0, width, height);
    var result = drawings[lessons[selected].kind]();
    if (announce) {
      $("sceneResult").textContent = result;
      $("stepCount").textContent = "步骤 " + (frame + 1) + " / " + frames();
      canvas.setAttribute("aria-label", lessons[selected].title + "。当前计算：" + result);
    }
  }
  function resize() {
    var bounds = canvas.getBoundingClientRect(), scale = Math.min(window.devicePixelRatio || 1, 2);
    width = bounds.width; height = bounds.height;
    canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
    context.setTransform(scale, 0, 0, scale, 0, 0);
    draw(true);
  }
  function pause() {
    playing = false;
    $("play").textContent = "▶ 播放动画";
  }
  function tick(timestamp) {
    if (!playing) return;
    if (!lastTick) lastTick = timestamp;
    fraction = Math.min(1, (timestamp - lastTick) / Number($("speed").value));
    if (fraction >= 1) {
      fraction = 0; lastTick = timestamp;
      if (frame >= frames() - 1) { pause(); draw(true); return; }
      frame++; draw(true);
    } else draw(false);
    requestAnimationFrame(tick);
  }
  function animate() {
    if (playing) { pause(); return; }
    if (frame >= frames() - 1) frame = 0;
    playing = true; lastTick = 0; fraction = 0;
    $("play").textContent = "Ⅱ 暂停动画";
    draw(true); requestAnimationFrame(tick);
  }
  function choose(index, scroll) {
    pause(); selected = Math.max(0, Math.min(lessons.length - 1, index)); frame = 0; fraction = 0;
    var lesson = lessons[selected];
    visited.add(lesson.id);
    try { localStorage.setItem("hush-math-visited", JSON.stringify(Array.from(visited))); } catch (_) {}
    history.replaceState(null, "", "#" + lesson.id);
    $("progress").textContent = "已读 " + visited.size + " / " + lessons.length;
    $("lessonNumber").textContent = lesson.category + " · LESSON " + String(selected + 1).padStart(2, "0");
    $("lessonTitle").textContent = lesson.title;
    $("lessonQuestion").textContent = lesson.question;
    $("lessonLocation").textContent = lesson.category.split(" / ")[1];
    $("sceneTitle").textContent = lesson.title + " · 数字可以自己验算";
    $("analogy").textContent = lesson.analogy;
    $("explanation").textContent = lesson.explanation;
    $("walkthrough").innerHTML = lesson.steps.map(function (step, index) {
      return "<div><b>" + String(index + 1).padStart(2, "0") + "</b><span>" + escape(step) + "</span></div>";
    }).join("");
    $("takeaway").textContent = lesson.takeaway;
    $("equations").innerHTML = lesson.equations.map(function (formula) { return '<div class="equation">' + escape(formula) + "</div>"; }).join("");
    $("symbols").innerHTML = lesson.symbols.map(function (symbol) { return "<dt>" + escape(symbol[0]) + "</dt><dd>" + escape(symbol[1]) + "</dd>"; }).join("");
    $("implementation").textContent = lesson.implementation;
    $("sources").innerHTML = lesson.sources.map(function (path) {
      return '<a href="https://github.com/johankoi91/AI_runtime/blob/main/' + escape(path) + '" target="_blank" rel="noopener">' + escape(path) + " ↗</a>";
    }).join("");
    parameters = {};
    $("parameters").innerHTML = lesson.controls.map(function (control) {
      parameters[control.id] = control.value;
      return '<div class="parameter"><label for="parameter-' + control.id + '">' + escape(control.label) +
        '<span id="value-' + control.id + '"></span></label>' + (control.options ?
        '<select id="parameter-' + control.id + '">' + control.options.map(function (option) {
          return '<option value="' + escape(option[0]) + '">' + escape(option[1]) + "</option>";
        }).join("") + "</select>" :
        '<input id="parameter-' + control.id + '" type="range" min="' + control.min + '" max="' + control.max +
        '" step="' + control.step + '" value="' + control.value + '">') + "</div>";
    }).join("");
    lesson.controls.forEach(function (control) {
      var input = $("parameter-" + control.id);
      input.value = control.value;
      function valueLabel() { $("value-" + control.id).textContent = control.options ? "" : input.value + (control.unit || ""); }
      valueLabel();
      input.addEventListener("input", function () {
        pause(); frame = 0; fraction = 0;
        parameters[control.id] = control.options ? input.value : Number(input.value);
        valueLabel(); draw(true);
      });
    });
    $("quizQuestion").textContent = lesson.quiz.question;
    $("quizFeedback").textContent = "先自己选一个答案，再看解释。";
    $("quizFeedback").className = "";
    $("quizOptions").innerHTML = lesson.quiz.choices.map(function (choice, index) {
      return '<button type="button" data-answer="' + index + '" aria-pressed="false">' + escape(choice) + "</button>";
    }).join("");
    $("quizOptions").querySelectorAll("button").forEach(function (button) {
      button.onclick = function () {
        var correct = Number(button.dataset.answer) === lesson.quiz.answer;
        $("quizOptions").querySelectorAll("button").forEach(function (other) { other.setAttribute("aria-pressed", String(other === button)); });
        $("quizFeedback").textContent = (correct ? "答对了。 " : "再想想：") + lesson.quiz.reason;
        $("quizFeedback").className = correct ? "correct" : "";
      };
    });
    $("chapters").querySelectorAll("button").forEach(function (button, index) {
      if (index === selected) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    $("previousLesson").disabled = selected === 0;
    $("nextLesson").textContent = selected === lessons.length - 1 ? "回到第一课 ↺" : "下一课 →";
    $("lessonPosition").textContent = (selected + 1) + " / " + lessons.length;
    resize();
    if (scroll) $("lesson").scrollIntoView({block:"start", behavior:matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"});
  }
  try {
    var saved = JSON.parse(localStorage.getItem("hush-math-visited") || "[]");
    if (Array.isArray(saved)) saved.forEach(function (id) { if (lessons.some(function (lesson) { return lesson.id === id; })) visited.add(id); });
  } catch (_) {}
  $("chapters").innerHTML = lessons.map(function (lesson, index) {
    return '<button type="button" data-lesson="' + index + '"><small>' + String(index + 1).padStart(2,"0") +
      " / " + escape(lesson.category.split(" / ")[1]) + "</small>" + escape(lesson.title) + "</button>";
  }).join("");
  $("chapters").querySelectorAll("button").forEach(function (button) {
    button.onclick = function () { choose(Number(button.dataset.lesson), true); };
  });
  $("formulaIndex").innerHTML = lessons.map(function (lesson, index) {
    return "<article><h3>" + String(index + 1).padStart(2,"0") + " · " + escape(lesson.title) +
      "</h3><pre>" + escape(lesson.equations.join("\n\n")) + "</pre></article>";
  }).join("");
  $("play").onclick = animate;
  $("step").onclick = function () { pause(); fraction = 0; frame = Math.min(frames() - 1, frame + 1); draw(true); };
  $("restart").onclick = function () { pause(); frame = 0; fraction = 0; draw(true); };
  $("previousLesson").onclick = function () { choose(selected - 1, true); };
  $("nextLesson").onclick = function () { choose((selected + 1) % lessons.length, true); };
  document.addEventListener("visibilitychange", function () { if (document.hidden) pause(); });
  window.addEventListener("hashchange", function () {
    var index = lessons.findIndex(function (lesson) { return lesson.id === location.hash.slice(1); });
    if (index >= 0 && index !== selected) choose(index, false);
  });
  choose(selected, false);
  new ResizeObserver(resize).observe(canvas);
}());
