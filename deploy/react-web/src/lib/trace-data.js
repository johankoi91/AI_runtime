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

function attributes(detail) {
  var values = {};
  var expression = /([a-z_]+)=(\[[^\]]*\]|[^\s]+)/gi;
  var match;
  while ((match = expression.exec(detail || ""))) values[match[1]] = match[2];
  return values;
}

function timeMs(event) {
  var time = event._time || "";
  var fraction = /\.(\d+)Z$/.exec(time);
  return Date.parse(time) + (fraction ? Number("0." + fraction[1]) * 1000 % 1 : 0);
}

function phaseFor(event) {
  var phases = {
    "hush.capture": "input",
    "hush.frontend": "frontend",
    "hush.encoder": "encoder",
    "hush.erb_decoder": "erb",
    "hush.df_decoder": "df",
    "hush.mask": "mask",
    "hush.deep_filter": "deep_filter",
    "hush.synthesis": "synthesis",
    "hush.state": "state"
  };
  if (event.stage === "hush.batch") return event.event === "stage_start" ? "input" : "output";
  return phases[event.stage] || null;
}

function batches(events) {
  var groups = [];
  events.forEach(function (event, index) {
    if (event.stage !== "hush.batch" || !/^stage_(start|complete|error)$/.test(event.event)) return;
    var metadata = attributes(event.detail);
    var key = [metadata.frame_start, metadata.frames].join("/");
    var group;
    if (event.event !== "stage_start") {
      group = groups.slice().reverse().find(function (item) { return item.key === key && !item.end; });
    }
    if (!group) {
      group = {
        id: String(event.sequence || index) + ":" + event._time,
        key: key,
        frameStart: metadata.frame_start !== undefined ? Number(metadata.frame_start) : null,
        frames: metadata.frames !== undefined ? Number(metadata.frames) : null,
        backend: metadata.backend || null,
        start: null, end: null, startIndex: null, endIndex: null
      };
      groups.push(group);
    }
    if (event.event === "stage_start") { group.start = event; group.startIndex = index; }
    else { group.end = event; group.endIndex = index; }
    if (metadata.backend) group.backend = metadata.backend;
  });
  groups.forEach(function (group, index) {
    var endIndex = group.endIndex === null ? events.length - 1 : group.endIndex;
    var nextStart = groups.slice(index + 1).find(function (item) { return item.startIndex !== null; });
    if (!group.end && nextStart) endIndex = nextStart.startIndex - 1;
    var begin = group.start ? timeMs(group.start) :
      timeMs(group.end) - Math.max(0, Number(group.end.duration_ms || 0));
    group.events = events.filter(function (event, eventIndex) {
      return eventIndex <= endIndex && (group.startIndex !== null
        ? eventIndex >= group.startIndex : timeMs(event) >= begin);
    });
    group.originMs = begin;
    group.complete = !!group.end && group.end.event === "stage_complete";
    group.error = !!group.end && group.end.event === "stage_error";
    group.incomplete = !group.start || !group.end;
    group.duration = group.complete && Number.isFinite(group.end.duration_ms) ? group.end.duration_ms : null;
  });
  if (!groups.length && events.some(function (event) { return phaseFor(event); })) {
    groups.push({
      id: "unscoped", frames: null, frameStart: null, backend: null, start: null, end: null,
      events: events, originMs: timeMs(events[0]), complete: false, error: false,
      incomplete: true, duration: null
    });
  }
  return groups;
}

function milestones(events, detailed) {
  if (detailed) return events.filter(function (event) {
    return phaseFor(event) && (/^stage_/.test(event.event) || event.event === "node_complete" ||
      event.event === "node_error" || event.event === "state_retained");
  });
  var firstStart = new Map(), lastComplete = new Map();
  events.forEach(function (event) {
    var phase = phaseFor(event);
    if (!phase) return;
    if (event.event === "stage_start" && !firstStart.has(phase)) firstStart.set(phase, event);
    if (event.event === "stage_complete") lastComplete.set(phase, event);
  });
  return events.filter(function (event) {
    var phase = phaseFor(event);
    return phase && (firstStart.get(phase) === event || lastComplete.get(phase) === event ||
      /_error$/.test(event.event) || event.event === "state_retained");
  });
}

function stageStats(events, phase) {
  var matches = events.filter(function (event) { return phaseFor(event) === phase; });
  var completed = matches.filter(function (event) { return event.event === "stage_complete"; });
  var nodes = matches.filter(function (event) { return event.event === "node_complete"; });
  var devices = Array.from(new Set(nodes.map(function (event) {
    return attributes(event.detail).device;
  }).filter(Boolean)));
  if (!devices.length) devices = Array.from(new Set(matches.map(function (event) {
    return attributes(event.detail).device;
  }).filter(Boolean)));
  var last = matches.length ? matches[matches.length - 1] : null;
  return {
    count: completed.length,
    duration: completed.length ? completed.reduce(function (total, event) {
      return total + Number(event.duration_ms || 0);
    }, 0) : null,
    devices: devices,
    nodes: nodes,
    last: last,
    output: completed.length ? completed[completed.length - 1] : null,
    error: matches.some(function (event) { return /_error$/.test(event.event); })
  };
}

var api = {
  parseCsv: parseCsv, runs: runs, attributes: attributes, timeMs: timeMs,
  phaseFor: phaseFor, batches: batches, milestones: milestones, stageStats: stageStats
};

export { parseCsv, runs, attributes, timeMs, phaseFor, batches, milestones, stageStats };
export default api;
