// The page: loads data/dashboard.json, draws the charts with Plotly, and runs the calculator.
// What it computes lives in logic.js and model.js; this file only reads controls and draws.
import {
  COLORS,
  MAP_CONTEXTS,
  MAP_USERS,
  fmt,
  inline,
  labelOf,
  mapAgreement,
  mdTable,
  predictionErrors,
  recommendationMap,
  serverRows,
  shortName,
  tokS,
  waterfall,
} from "./logic.js";
import { dollarsPerMillion, kvCapacity, makeLoad, makeStack, predict, runChecks } from "./model.js";

const $ = (id) => document.getElementById(id);
const PLOT = { displayModeBar: false, responsive: true };
const FONT = { family: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size: 13, color: "#1c1c1c" };
const layout = (extra) => ({ font: FONT, margin: { l: 64, r: 18, t: 12, b: 64 }, ...extra });
const MAP_COLORS = {
  base: "#6E6E6E",
  w: "#9ecae1",
  wk: "#3182bd",
  ws: "#fdae6b",
  wks: "#e6550d",
  a: "#c7e9c0",
  ak: "#31a354",
  as: "#dadaeb",
  aks: "#756bb1",
};

function fillSelect(select, options, selected) {
  select.innerHTML = options.map(([value, text]) => `<option value="${value}">${text}</option>`).join("");
  if (selected != null) select.value = selected;
}

function workloadName(data, workload) {
  return data.workloads[workload].label;
}

// ---- 1. the waterfall ----------------------------------------------------------------------------------

function drawWaterfall(data) {
  const model = $("waterfall-model").value;
  const workload = $("waterfall-workload").value;
  const unit = $("waterfall-unit").value;
  const fall = waterfall(data, model, workload, unit);
  if (!fall) {
    Plotly.purge("waterfall");
    $("waterfall-takeaway").textContent = "This model's ladder was not run on this workload.";
    return;
  }
  const names = ["stock BF16", ...fall.steps.map((s) => `+ ${s.name}`), "full stack"];
  const bases = [0, ...fall.steps.map((s) => Math.min(s.before, s.after)), 0];
  const heights = [fall.base, ...fall.steps.map((s) => Math.abs(s.after - s.before)), fall.full];
  const colors = [COLORS.base, ...fall.steps.map((s) => COLORS[s.letter]), COLORS.full];
  const shapes = ["", ...fall.steps.map((s) => (s.worse ? "x" : "")), ""];
  const text = ["", ...fall.steps.map((s) => `${s.change >= 0 ? "+" : ""}${(100 * s.change).toFixed(0)}%`), ""];
  if (fall.best) {
    names.push(`best: ${fall.best.label}`);
    bases.push(0);
    heights.push(fall.best.value);
    colors.push(COLORS.best);
    shapes.push("");
    text.push("");
  }
  const digits = unit === "dollars" ? 2 : 0;
  const trace = {
    type: "bar",
    x: names,
    y: heights,
    base: bases,
    text,
    textposition: "outside",
    marker: { color: colors, pattern: { shape: shapes, fgcolor: "#1c1c1c" } },
    customdata: heights.map((h, i) => bases[i] + h),
    hovertemplate: `%{x}<br>%{customdata:,.${digits}f}<extra></extra>`,
  };
  const title = unit === "dollars" ? "$ per 1M output tokens" : "output tokens per second";
  Plotly.react("waterfall", [trace], layout({ yaxis: { title, rangemode: "tozero" } }), PLOT);
  const best = fall.bestLabel ? `the best measured stack is <code>${fall.bestLabel}</code> at ${fall.bestGain.toFixed(2)}×` : "";
  const full = `the full stack gives ${fall.fullGain.toFixed(2)}× stock`;
  const verdict = fall.best ? "It is not the full stack." : "It is the full stack.";
  $("waterfall-takeaway").innerHTML = `${shortName(model)}, ${workloadName(data, workload)}: ${full}; ${best}. ${verdict}`;
}

// ---- 2. every server -----------------------------------------------------------------------------------

function drawServers(data) {
  const model = $("servers-model").value;
  const workload = $("servers-workload").value;
  const rows = serverRows(data, model, workload, $("servers-controls").checked);
  const color = (row) => (!row.deployable ? "#b8b8b8" : row.graph === "piecewise" ? COLORS.s : COLORS.w);
  const trace = {
    type: "bar",
    x: rows.map((r) => r.label),
    y: rows.map((r) => r.gain),
    marker: { color: rows.map(color) },
    customdata: rows.map((r) => [r.tok_s, r.tpot_ms, r.ttft_ms, r.power_w, r.kept, r.graph]),
    hovertemplate:
      "<b>%{x}</b>: %{y:.2f}× stock<br>%{customdata[0]:,.0f} tokens/s<br>TPOT %{customdata[1]:.1f} ms, " +
      "TTFT %{customdata[2]:,.0f} ms<br>GPU %{customdata[3]:.0f} W, %{customdata[4]:.2f} tokens per pass<br>" +
      "CUDA graphs: %{customdata[5]}<extra></extra>",
  };
  const line = { type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: "#6E6E6E", dash: "dot", width: 1 } };
  Plotly.react(
    "servers-chart",
    [trace],
    layout({ yaxis: { title: "tokens/s ÷ the stock server's" }, xaxis: { type: "category" }, shapes: [line] }),
    PLOT,
  );
  const deployable = rows.filter((r) => r.deployable);
  const top = deployable[0];
  const piecewise = deployable.filter((r) => r.graph === "piecewise");
  const worst = piecewise.length ? piecewise[piecewise.length - 1] : null;
  let text = `Fastest: <code>${top.label}</code> at ${top.gain.toFixed(2)}× (${fmt(top.tok_s)} tokens/s). `;
  text += "Blue servers keep their full CUDA graph; orange ones run on piecewise graphs";
  text += worst ? ` (the slowest of them, <code>${worst.label}</code>, is at ${worst.gain.toFixed(2)}×).` : ".";
  $("servers-takeaway").innerHTML = text;
}

// ---- 3. interactions -----------------------------------------------------------------------------------

function drawInteractions(data) {
  const workloads = data.interaction_workloads;
  const z = data.interactions.map((row) => workloads.map((w) => row[w]));
  const flat = z.flat().filter((v) => v != null);
  const span = Math.max(Math.max(...flat), 1 / Math.min(...flat), 1.2);
  const trace = {
    type: "heatmap",
    x: workloads.map((w) => workloadName(data, w).replace(" (", "<br>(")),
    y: data.interactions.map((row) => row.pair),
    z: z.map((row) => row.map((v) => (v == null ? null : Math.log(v)))),
    text: z.map((row) => row.map((v) => (v == null ? "" : v.toFixed(2)))),
    texttemplate: "%{text}",
    colorscale: "PuOr",
    reversescale: true,
    zmin: -Math.log(span),
    zmax: Math.log(span),
    showscale: false,
    hovertemplate: "%{y}<br>%{x}<br>combined ÷ product: %{text}<extra></extra>",
  };
  Plotly.react("interaction-chart", [trace], layout({ margin: { l: 300, r: 18, t: 12, b: 70 }, xaxis: { tickangle: 0 }, yaxis: { autorange: "reversed" } }), PLOT);
  const near = flat.filter((v) => Math.abs(v - 1) <= 0.05).length;
  let worst = { value: Infinity };
  data.interactions.forEach((row) =>
    workloads.forEach((w) => {
      if (row[w] != null && row[w] < worst.value) worst = { value: row[w], pair: row.pair, workload: w };
    }),
  );
  $("interaction-takeaway").textContent =
    `${near} of ${flat.length} pairs multiply to within 5%. The pair that competes most is ${worst.pair} on ` +
    `${workloadName(data, worst.workload)} (${worst.value.toFixed(2)}).`;
}

// ---- 4. the host's chain -------------------------------------------------------------------------------

function drawHost(data) {
  const rows = [...data.host_chain].sort((a, b) => (a.model === b.model ? a.full_ms - b.full_ms : a.model < b.model ? -1 : 1));
  const x = rows.map((r) => `${shortName(r.model).replace("Qwen3-", "")} ${r.label}<br>vs ${r.twin}`);
  const full = { type: "bar", name: "closest server with a full CUDA graph", x, y: rows.map((r) => r.full_ms), marker: { color: COLORS.base } };
  const piecewise = {
    type: "bar",
    name: "on piecewise graphs",
    x,
    y: rows.map((r) => r.piecewise_ms),
    marker: { color: COLORS.s },
    customdata: rows.map((r) => r.power_w),
    hovertemplate: "%{y:.1f} ms per step<br>GPU power %{customdata:.0f} W<extra></extra>",
  };
  Plotly.react(
    "host-chart",
    [full, piecewise],
    layout({ barmode: "group", yaxis: { title: "ms per engine step, one user" }, legend: { orientation: "h", y: 1.12 }, margin: { l: 64, r: 18, t: 30, b: 90 } }),
    PLOT,
  );
  const slower = rows.filter((r) => r.piecewise_ms > 1.15 * r.full_ms);
  const worst = rows.reduce((a, b) => (b.piecewise_ms / b.full_ms > a.piecewise_ms / a.full_ms ? b : a));
  $("host-takeaway").innerHTML =
    `${slower.length} of ${rows.length} servers on piecewise graphs are slower than their full-graph twin, by up to ` +
    `${(worst.piecewise_ms / worst.full_ms).toFixed(1)}× (<code>${worst.label}</code> on ${shortName(worst.model)}: ` +
    `${worst.piecewise_ms.toFixed(0)} ms per step against ${worst.full_ms.toFixed(0)}). The rest are the ones whose GPU ` +
    "already needs longer per step than the host does.";
}

// ---- 5. predicted against measured ---------------------------------------------------------------------

function drawPredicted(data) {
  const errors = predictionErrors(data);
  const styles = {
    "no speculation": { color: "#000000", symbol: "circle" },
    speculation: { color: COLORS.s, symbol: "circle" },
    "FP8 KV + speculation": { color: COLORS.s, symbol: "circle-open" },
  };
  const traces = Object.entries(styles).map(([group, style]) => {
    const rows = errors.rows.filter((r) => r.group === group);
    return {
      type: "scatter",
      mode: "markers",
      name: group,
      x: rows.map((r) => r.measured),
      y: rows.map((r) => r.tok_s),
      marker: { color: style.color, symbol: style.symbol, size: 9, line: { width: 1.5, color: style.color } },
      customdata: rows.map((r) => [shortName(r.model), r.label, workloadName(data, r.workload), 100 * r.error]),
      hovertemplate:
        "%{customdata[0]} <b>%{customdata[1]}</b><br>%{customdata[2]}<br>measured %{x:,.0f}, predicted %{y:,.0f} " +
        "tokens/s (%{customdata[3]:+.0f}%)<extra></extra>",
    };
  });
  const all = errors.rows.flatMap((r) => [r.measured, r.tok_s]);
  const [lo, hi] = [Math.min(...all) * 0.7, Math.max(...all) * 1.4];
  const band = (factor, extra) => ({ type: "scatter", mode: "lines", x: [lo, hi], y: [lo * factor, hi * factor], hoverinfo: "skip", showlegend: false, line: { color: "#b8b8b8", width: 1 }, ...extra });
  const lines = [band(0.85), band(1.15, { fill: "tonexty", fillcolor: "rgba(110,110,110,0.15)" }), band(1)];
  Plotly.react(
    "predicted-chart",
    [...lines, ...traces],
    layout({
      xaxis: { type: "log", dtick: 1, title: "measured (output tokens/s)" },
      yaxis: { type: "log", dtick: 1, title: "predicted before the run (output tokens/s)" },
      legend: { orientation: "h", y: 1.08 },
      margin: { l: 70, r: 18, t: 30, b: 60 },
    }),
    PLOT,
  );
  const pct = (v) => (v == null ? "—" : `${(100 * v).toFixed(0)}%`);
  $("predicted-takeaway").textContent =
    `${errors.rows.length} predictions, median error ${pct(errors.all)} (${pct(errors.within)} within 15%, the grey band): ` +
    `${pct(errors.groups["no speculation"])} without speculation, ${pct(errors.groups.speculation)} with it, and ` +
    `${pct(errors.groups["FP8 KV + speculation"])} where FP8 KV meets speculation (hollow markers), which the model had no term for.`;
}

// ---- 6. the calculator ---------------------------------------------------------------------------------

function calculatorInputs() {
  return {
    model: $("calc-model").value,
    users: Math.max(1, Number($("calc-users").value) || 1),
    prompt: Math.max(1, Number($("calc-prompt").value) || 1),
    output: Math.max(1, Number($("calc-output").value) || 1),
    weights: $("calc-weights").value,
    fp8Kv: $("calc-kv").value === "fp8",
    spec: $("calc-spec").checked,
    kept: Math.max(1, Number($("calc-kept").value) || 1),
    prefix: $("calc-prefix").checked,
    cached: Math.min(100, Math.max(0, Number($("calc-cached").value) || 0)) / 100,
    calibration: $("calc-calibration").value,
  };
}

function tile(name, value, warn = false) {
  return `<div class="tile${warn ? " warn" : ""}"><div class="value">${value}</div><div class="name">${name}</div></div>`;
}

function drawCalculator(data) {
  const input = calculatorInputs();
  const cfg = data.models[input.model];
  const cal = data.calibrations[input.calibration];
  const preset = data.workloads[$("calc-preset").value]?.load;
  const speculation = input.spec ? { k: 3, tokens_per_pass: input.kept, draft_bytes: data.draft_bytes[input.model] } : null;
  const stack = makeStack({ weights: input.weights, fp8Kv: input.fp8Kv, prefix: input.prefix, speculation });
  const stock = makeStack({});
  const loadFor = (s) =>
    makeLoad({
      users: preset ? preset.users : input.users,
      prompt_len: preset ? preset.prompt_len : input.prompt,
      output_len: preset ? preset.output_len : input.output,
      cached_len: preset ? preset.cached_len : input.cached * input.prompt,
      context: preset ? preset.context : null,
      shared_len: preset ? preset.shared_len : input.cached * input.prompt,
      sharers: preset ? preset.sharers : input.prefix && input.cached > 0 ? 2 : 1,
      kv_tokens: kvCapacity(cfg, s, data.startup_memory[input.model]),
    });
  const got = predict(cfg, data.hardware, stack, cal, loadFor(stack));
  const base = predict(cfg, data.hardware, stock, cal, loadFor(stock));
  const users = preset ? preset.users : input.users;
  const limited = got.batch < Math.min(users, 256) - 0.5;
  $("calc-tiles").innerHTML = [
    tile("output tokens per second", fmt(got.tok_s)),
    tile("× the stock server at this load", `${(got.tok_s / base.tok_s).toFixed(2)}×`),
    tile("ms between a user's tokens (TPOT)", fmt(got.tpot_ms, 1)),
    tile("ms to the first token (one user)", got.ttft_ms == null ? "—" : fmt(got.ttft_ms)),
    tile("sequences running at once", fmt(got.batch), limited),
    tile("$ per 1M output tokens", fmt(dollarsPerMillion(got.tok_s, data.price_per_hour), 2)),
    tile("of a decode step is the KV read", `${(100 * got.kv_share).toFixed(0)}%`),
  ].join("");
  const notes = [];
  if (limited) notes.push(`The KV cache holds ${fmt(got.batch)} of these ${fmt(users)} requests at once; the rest wait in the queue.`);
  if (got.host_bound) notes.push("An FP8 KV cache with speculation loses the full CUDA graph on this GPU, and at this load the pass waits for the host, not the GPU.");
  if (input.calibration === "frozen" && stack.speculation && stack.backend === "flashinfer") notes.push("The constants frozen before M8 have no host term: this is the prediction that M8 proved wrong at low load.");
  if (input.weights === "int4") notes.push("INT4 weights cost visibly more quality than FP8 (M4).");
  $("calc-notes").textContent = notes.join(" ");

  const label = labelOf(input);
  const measured = Object.keys(data.workloads)
    .map((w) => ({ workload: w, rate: tokS(data, input.model, label, w), base: tokS(data, input.model, "base", w) }))
    .filter((row) => row.rate != null);
  if (measured.length === 0) {
    $("calc-measured").innerHTML = `<p class="note">The combination <code>${label}</code> was not run on ${shortName(input.model)}.</p>`;
  } else {
    const rows = measured.map((r) => `| ${workloadName(data, r.workload)} | ${fmt(r.rate)} | ${(r.rate / r.base).toFixed(2)}× |`);
    $("calc-measured").innerHTML =
      `<p class="note">Server <code>${label}</code> on ${shortName(input.model)}, as measured:</p>` +
      mdTable(["| Workload | Tokens/s | vs stock |", "|---|---|---|", ...rows].join("\n"));
  }
  drawMap(data, input);
}

function drawMap(data, input) {
  const allowInt4 = $("map-int4").checked;
  const grid = recommendationMap(data, input.model, input.calibration, input.output, input.kept, allowInt4);
  const labels = Object.keys(MAP_COLORS);
  const { stars, checked } = allowInt4 ? { stars: [], checked: 0 } : mapAgreement(data, input.model, grid);
  const isStar = (u, c) => stars.some(([su, sc]) => su === u && sc === c);
  const scale = labels.flatMap((label, i) => [
    [i / labels.length, MAP_COLORS[label]],
    [(i + 1) / labels.length, MAP_COLORS[label]],
  ]);
  const trace = {
    type: "heatmap",
    x: MAP_CONTEXTS.map((c) => c.toLocaleString("en-US")),
    y: MAP_USERS.map(String),
    z: grid.map((row) => row.map((cell) => labels.indexOf(cell.label) + 0.5)),
    text: grid.map((row, i) => row.map((cell, j) => `<b>${cell.label}</b>${isStar(MAP_USERS[i], MAP_CONTEXTS[j]) ? " ★" : ""}<br>${cell.gain.toFixed(1)}×`)),
    texttemplate: "%{text}",
    colorscale: scale,
    zmin: 0,
    zmax: labels.length,
    showscale: false,
    xgap: 2,
    ygap: 2,
    hovertemplate: "%{y} users, %{x} tokens of context<br>%{text}<extra></extra>",
  };
  Plotly.react(
    "map-chart",
    [trace],
    layout({ xaxis: { title: "context per request (tokens)", type: "category" }, yaxis: { title: "concurrent users", type: "category" } }),
    PLOT,
  );
  const counts = {};
  grid.flat().forEach((cell) => (counts[cell.label] = (counts[cell.label] || 0) + 1));
  const total = grid.flat().length;
  const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 3);
  let text = `${shortName(input.model)}: ` + ranked.map(([label, n]) => `<code>${label}</code> in ${((100 * n) / total).toFixed(0)}% of the cells`).join(", ") + ".";
  if (checked) text += ` The pick is the fastest measured stack on ${stars.length} of the ${checked} measured workloads (★); everywhere else the map is a prediction.`;
  $("map-takeaway").innerHTML = text;
}

function applyPreset(data) {
  const load = data.workloads[$("calc-preset").value]?.load;
  if (!load) return;
  $("calc-users").value = Math.round(load.users);
  $("calc-prompt").value = Math.round(load.prompt_len);
  $("calc-output").value = Math.round(load.output_len);
  $("calc-cached").value = Math.round((100 * load.cached_len) / load.prompt_len);
}

// ---- 7. needle grids -----------------------------------------------------------------------------------

function drawNeedle(data) {
  const grid = data.needles[Number($("needle-select").value)];
  const trace = {
    type: "heatmap",
    x: grid.lengths.map((n) => n.toLocaleString("en-US")),
    y: grid.depths.map((d) => `${(100 * d).toFixed(0)}%`),
    z: grid.passed,
    text: grid.passed.map((row) => row.map((v) => `${(100 * v).toFixed(0)}%`)),
    texttemplate: "%{text}",
    colorscale: [[0, "#D55E00"], [0.5, "#F0E442"], [1, "#009E73"]],
    zmin: 0,
    zmax: 1,
    showscale: false,
    xgap: 2,
    ygap: 2,
    hovertemplate: "prompt of %{x} tokens, needle at %{y} depth<br>%{text} of secrets recalled<extra></extra>",
  };
  Plotly.react(
    "needle-chart",
    [trace],
    layout({ xaxis: { title: "prompt length (tokens)", type: "category" }, yaxis: { title: "needle depth in the prompt", type: "category" } }),
    PLOT,
  );
  const cells = grid.passed.flat();
  const recall = cells.reduce((a, b) => a + b, 0) / cells.length;
  $("needle-takeaway").innerHTML = `${inline(grid.title)}: ${(100 * recall).toFixed(0)}% of the secrets recalled over ${cells.length} cells.`;
}

// ---- start ---------------------------------------------------------------------------------------------

async function main() {
  const data = await (await fetch("data/dashboard.json")).json();
  const env = data.environment;
  $("environment").textContent = `Qwen3-0.6B and Qwen3-1.7B on one ${env.gpu}, vLLM ${env.vllm}, PyTorch ${env.torch}.`;
  $("findings").innerHTML = data.findings.map((line) => `<li>${inline(line.replace(/^- /, ""))}</li>`).join("");

  const models = Object.keys(data.models).map((m) => [m, shortName(m)]);
  const workloads = Object.keys(data.workloads).map((w) => [w, data.workloads[w].label]);
  for (const id of ["waterfall-model", "servers-model", "calc-model"]) fillSelect($(id), models);
  for (const id of ["waterfall-workload", "servers-workload"]) fillSelect($(id), workloads);
  $("calc-preset").innerHTML += workloads.map(([w, text]) => `<option value="${w}">${text}</option>`).join("");
  fillSelect($("needle-select"), data.needles.map((grid, i) => [i, grid.title.replace(/`/g, "")]), data.needles.length - 1);
  $("calc-kept").value = data.tokens_per_pass.toFixed(2);
  $("table-list").innerHTML = data.tables.map((t) => `<details><summary>${inline(t.title)}</summary>${mdTable(t.markdown)}</details>`).join("");

  const bind = (ids, draw) => ids.forEach((id) => $(id).addEventListener("input", () => draw(data)));
  bind(["waterfall-model", "waterfall-workload", "waterfall-unit"], drawWaterfall);
  bind(["servers-model", "servers-workload", "servers-controls"], drawServers);
  bind(["needle-select"], drawNeedle);
  $("calc-preset").addEventListener("input", () => {
    applyPreset(data);
    drawCalculator(data);
  });
  for (const element of $("calc-form").elements) {
    if (element.id !== "calc-preset") {
      element.addEventListener("input", () => {
        if (["calc-users", "calc-prompt", "calc-output", "calc-cached"].includes(element.id)) $("calc-preset").value = "";
        drawCalculator(data);
      });
    }
  }
  $("map-int4").addEventListener("input", () => drawCalculator(data));

  drawWaterfall(data);
  drawServers(data);
  drawInteractions(data);
  drawHost(data);
  drawPredicted(data);
  drawCalculator(data);
  drawNeedle(data);

  const failures = runChecks(data);
  const parity = $("parity");
  if (failures.length === 0) {
    parity.className = "ok";
    parity.textContent = `The calculator's model reproduces the Python model's numbers on all ${data.checks.length} checks shipped with this page.`;
  } else {
    parity.className = "bad";
    parity.textContent = `The calculator disagrees with the Python model on ${failures.length} values: do not trust it.`;
  }
}

main().catch((error) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="note">Could not load the results: ${error}. Serve this folder over HTTP (python -m http.server -d site).</p>`);
});
