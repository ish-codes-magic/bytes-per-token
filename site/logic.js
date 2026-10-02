// Everything the page computes that does not touch the DOM or Plotly, so Node can test it
// (site/tests/logic.mjs).
import { dollarsPerMillion, kvCapacity, makeLoad, makeStack, predict } from "./model.js";

export const LARGE = "Qwen/Qwen3-1.7B";
export const TECHNIQUE_ORDER = "wafkpsg"; // the order of letters in a server's label
export const COLORS = {
  base: "#6E6E6E",
  w: "#0072B2", // FP8 weights
  a: "#E69F00", // INT4 weights
  k: "#009E73", // FP8 KV cache
  p: "#CC79A7", // prefix caching
  s: "#D55E00", // speculative decoding
  full: "#000000",
  best: "#56B4E9",
};

// label -> [weights, FP8 KV, speculation]: what the model is asked to choose from (as in report/m8.py)
export const CANDIDATE_STACKS = {
  base: ["bf16", false, false],
  w: ["fp8", false, false],
  wk: ["fp8", true, false],
  ws: ["fp8", false, true],
  wks: ["fp8", true, true],
  a: ["int4", false, false],
  ak: ["int4", true, false],
  as: ["int4", false, true],
  aks: ["int4", true, true],
};
export const MAP_USERS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
export const MAP_CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768];
export const MEASURED_AT = {
  // workload -> [users, context] of the map cell closest to it
  m8_latency: [1, 512],
  spec_mixed: [64, 512],
  capacity: [64, 4096],
  long_32k: [1, 32768],
};

export function shortName(model) {
  return model.split("/").pop();
}

export function escapeHtml(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// The little Markdown the generated text uses: **bold**, `code`, [text](link).
export function inline(markdown) {
  return escapeHtml(markdown)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

// A Markdown table (as scripts/render_docs.py writes them) as an HTML table.
export function mdTable(markdown) {
  const rows = markdown
    .split("\n")
    .filter((line) => line.trim().startsWith("|"))
    .map((line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
  if (rows.length < 2) return `<p>${inline(markdown)}</p>`;
  const [head, , ...body] = rows; // the second row is the |---| separator
  const th = head.map((cell) => `<th>${inline(cell)}</th>`).join("");
  const trs = body.map((row) => `<tr>${row.map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table></div>`;
}

// The label of a stack from the calculator's switches: letters in the plan's order, or "base".
export function labelOf({ weights = "bf16", fp8Kv = false, prefix = false, spec = false }) {
  const on = new Set();
  if (weights === "fp8") on.add("w");
  if (weights === "int4") on.add("a");
  if (fp8Kv) on.add("k");
  if (prefix) on.add("p");
  if (spec) on.add("s");
  return [...TECHNIQUE_ORDER].filter((letter) => on.has(letter)).join("") || "base";
}

export function tokS(data, model, label, workload) {
  return data.servers[model]?.[label]?.workloads?.[workload]?.tok_s ?? null;
}

// One workload down the ladder, in $ per 1M tokens or tokens/s: the base, one step per technique, the full
// stack, and the best measured stack when it is a different one.
export function waterfall(data, model, workload, unit = "dollars") {
  const value = (label) => {
    const rate = tokS(data, model, label, workload);
    if (rate == null) return null;
    return unit === "dollars" ? dollarsPerMillion(rate, data.price_per_hour) : rate;
  };
  const values = data.ladder.map(value);
  if (values.includes(null)) return null;
  // The technique a ladder step adds is the letter its label gains: base -> w -> wk -> wkp -> wkps.
  const added = (i) => [...data.ladder[i + 1]].find((letter) => i === 0 || !data.ladder[i].includes(letter));
  const steps = data.ladder.slice(1).map((_, i) => {
    const letter = added(i);
    const [before, after] = [values[i], values[i + 1]];
    const worse = unit === "dollars" ? after > before : after < before;
    return { letter, name: data.techniques[letter], before, after, change: after / before - 1, worse };
  });
  const top = data.best[model]?.[workload]?.fp8;
  const fullLabel = data.ladder[data.ladder.length - 1];
  const best = top && top[0] !== fullLabel ? { label: top[0], value: value(top[0]) } : null;
  const baseRate = tokS(data, model, "base", workload);
  return {
    base: values[0],
    steps,
    full: values[values.length - 1],
    best,
    bestLabel: top ? top[0] : null,
    bestGain: top ? top[1] / baseRate : null,
    fullGain: tokS(data, model, fullLabel, workload) / baseRate,
  };
}

// Every measured server of a model on one workload, fastest first.
export function serverRows(data, model, workload, withControls = false) {
  const base = tokS(data, model, "base", workload);
  const rows = [];
  for (const [label, server] of Object.entries(data.servers[model])) {
    const measured = server.workloads[workload];
    if (!measured || label.endsWith("-r2")) continue;
    if (!server.deployable && !withControls) continue;
    rows.push({ label, gain: measured.tok_s / base, ...measured, graph: server.graph, deployable: server.deployable });
  }
  return rows.sort((a, b) => b.gain - a.gain);
}

function candidateStack(data, model, label, kept) {
  const [weights, fp8Kv, spec] = CANDIDATE_STACKS[label];
  const speculation = spec ? { k: 3, tokens_per_pass: kept, draft_bytes: data.draft_bytes[model] } : null;
  return makeStack({ weights, fp8Kv, speculation });
}

// The model's prediction for every candidate stack at one (users, context per request).
export function predictCandidates(data, model, calibration, users, context, outputLen, kept) {
  const cfg = data.models[model];
  const out = {};
  for (const label of Object.keys(CANDIDATE_STACKS)) {
    const stack = candidateStack(data, model, label, kept);
    const load = makeLoad({
      users,
      prompt_len: context,
      output_len: outputLen,
      kv_tokens: kvCapacity(cfg, stack, data.startup_memory[model]),
    });
    out[label] = predict(cfg, data.hardware, stack, data.calibrations[calibration], load);
  }
  return out;
}

export function pickStack(predictions, allowInt4) {
  let winner = null;
  for (const [label, got] of Object.entries(predictions)) {
    if (!allowInt4 && label.startsWith("a")) continue;
    if (winner == null || got.tok_s > predictions[winner].tok_s) winner = label;
  }
  return { label: winner, gain: predictions[winner].tok_s / predictions.base.tok_s };
}

// The recommendation map: for each (users, context) cell the model's pick and its gain over stock.
export function recommendationMap(data, model, calibration, outputLen, kept, allowInt4) {
  return MAP_USERS.map((users) =>
    MAP_CONTEXTS.map((context) =>
      pickStack(predictCandidates(data, model, calibration, users, context, outputLen, kept), allowInt4),
    ),
  );
}

// Where the map's pick is also the fastest measured stack: [[users, context], ...] and how many were checked.
export function mapAgreement(data, model, grid) {
  const stars = [];
  let checked = 0;
  for (const [workload, [users, context]] of Object.entries(MEASURED_AT)) {
    const top = data.best[model]?.[workload]?.fp8;
    if (!top) continue;
    checked += 1;
    const measured = top[0].replace("p", "") || "base"; // the map has no prefix caching: it never hurts
    const pick = grid[MAP_USERS.indexOf(users)][MAP_CONTEXTS.indexOf(context)].label;
    if (pick === measured) stars.push([users, context]);
  }
  return { stars, checked };
}

export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function predictionGroup(label) {
  const letters = label.split("-")[0];
  if (letters !== "base" && letters.includes("s") && letters.includes("k")) return "FP8 KV + speculation";
  return letters !== "base" && letters.includes("s") ? "speculation" : "no speculation";
}

// Median absolute error of the frozen predictions, overall and per group.
export function predictionErrors(data) {
  const errors = data.predictions.map((p) => ({ ...p, error: p.tok_s / p.measured - 1, group: predictionGroup(p.label) }));
  const by = (group) => errors.filter((e) => e.group === group).map((e) => Math.abs(e.error));
  return {
    rows: errors,
    all: median(errors.map((e) => Math.abs(e.error))),
    within: errors.filter((e) => Math.abs(e.error) <= 0.15).length / errors.length,
    groups: Object.fromEntries(
      ["no speculation", "speculation", "FP8 KV + speculation"].map((g) => [g, by(g).length ? median(by(g)) : null]),
    ),
  };
}

export function fmt(value, digits = 0) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
