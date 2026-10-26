(function (root) {
  "use strict";

  function parseCsv(text) {
    var rows = [], row = [], field = "", quoted = false;
    for (var index = 0; index < text.length; index += 1) {
      var character = text[index];
      if (character === '"') {
        if (quoted && text[index + 1] === '"') { field += '"'; index += 1; }
        else quoted = !quoted;
      } else if (character === "," && !quoted) {
        row.push(field); field = "";
      } else if ((character === "\n" || character === "\r") && !quoted) {
        if (character === "\r" && text[index + 1] === "\n") index += 1;
        row.push(field); rows.push(row); row = []; field = "";
      } else field += character;
    }
    if (field || row.length) { row.push(field); rows.push(row); }
    var header = [], grouped = new Map();
    rows.forEach(function (values) {
      if (!values.length || values[0].charAt(0) === "#") return;
      if (values.indexOf("_time") >= 0 && values.indexOf("run_id") >= 0) {
        header = values;
        return;
      }
      if (!header.length) return;
      var item = {};
      header.forEach(function (key, index) { if (key) item[key] = values[index] || ""; });
      if (!Number.isFinite(Date.parse(item._time)) || !item.run_id || !item._field) return;
      var key = [item._time, item.run_id, item.stage || ""].join("\u0000");
      if (!grouped.has(key)) grouped.set(key, {
        _time: item._time, run_id: item.run_id, stage: item.stage || ""
      });
      if (["event", "detail", "sequence", "bytes", "duration_ms"].indexOf(item._field) >= 0)
        grouped.get(key)[item._field] = item._value;
    });
    var result = Array.from(grouped.values()).filter(function (item) { return item.event; });
    result.forEach(function (item) {
      item.sequence = Number(item.sequence || 0);
      item.bytes = Number(item.bytes || 0);
      item.duration_ms = Number(item.duration_ms || 0);
    });
    return result.sort(function (left, right) {
      var normalize = function (time) {
        return time.replace(/(\.\d+)?Z$/, function (_, fraction) {
          return "." + (fraction || ".").slice(1).padEnd(9, "0") + "Z";
        });
      };
      return normalize(left._time).localeCompare(normalize(right._time)) || left.sequence - right.sequence;
    });
  }

  function runs(events) {
    var grouped = new Map();
    events.forEach(function (event) {
      if (!grouped.has(event.run_id)) grouped.set(event.run_id, []);
      grouped.get(event.run_id).push(event);
    });
    return Array.from(grouped, function (entry) {
      var hush = entry[1].some(function (event) { return event.stage.indexOf("hush.") === 0; });
      var old = !hush && entry[1].some(function (event) {
        return /metal demo|matmul_kernel|add_kernel/.test(event.detail || "");
      });
      return { id: entry[0], events: entry[1], kind: hush ? "hush" : old ? "old" : "unknown" };
    }).sort(function (left, right) {
      return Date.parse(right.events[right.events.length - 1]._time) -
        Date.parse(left.events[left.events.length - 1]._time);
    });
  }

  var api = { parseCsv: parseCsv, runs: runs };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RuntimeTraceData = api;
}(typeof window !== "undefined" ? window : this));
