// Tests of what the page computes, against the data file and against Python's own answers in it.
//   node site/tests/logic.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  CANDIDATE_STACKS,
  LARGE,
  MAP_CONTEXTS,
  MAP_USERS,
  inline,
  labelOf,
  mapAgreement,
  mdTable,
  predictionErrors,
  recommendationMap,
  serverRows,
  tokS,
  waterfall,
} from "../logic.js";
import { kvCapacity, makeLoad, makeStack, predict } from "../model.js";

const here = dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(readFileSync(join(here, "..", "data", "dashboard.json"), "utf-8"));
const close = (a, b, tolerance = 1e-5) => Math.abs(a - b) <= tolerance * Math.abs(b);
let passed = 0;
function test(name, body) {
  body();
  passed += 1;
}

test("labels follow the plan's letter order", () => {
  assert.equal(labelOf({}), "base");
  assert.equal(labelOf({ weights: "fp8", fp8Kv: true, prefix: true, spec: true }), "wkps");
  assert.equal(labelOf({ weights: "int4", prefix: true, spec: true }), "aps");
  assert.equal(labelOf({ spec: true, fp8Kv: true }), "ks");
});

test("markdown: bold, code, links and tables, with HTML escaped", () => {
  assert.equal(inline("**a** `b<c>`"), "<strong>a</strong> <code>b&lt;c&gt;</code>");
  assert.equal(inline("[x](https://e.org)"), '<a href="https://e.org">x</a>');
  const table = mdTable("| A | B |\n|---|---|\n| 1 | `w` |");
  assert.ok(table.includes("<th>A</th><th>B</th>") && table.includes("<td>1</td><td><code>w</code></td>"));
  assert.equal(mdTable("*nothing ran*"), "<p>*nothing ran*</p>");
});

test("the waterfall's ends are the measured servers", () => {
  const fall = waterfall(data, LARGE, "m8_latency", "tok_s");
  assert.deepEqual(fall.steps.map((step) => step.letter), ["w", "k", "p", "s"]); // the order they were added
  assert.deepEqual(fall.steps.map((step) => step.name), ["w", "k", "p", "s"].map((x) => data.techniques[x]));
  fall.steps.forEach((step, i) => {
    assert.ok(close(step.before, tokS(data, LARGE, data.ladder[i], "m8_latency")));
    assert.ok(close(step.after, tokS(data, LARGE, data.ladder[i + 1], "m8_latency")));
  });
  assert.ok(close(fall.base, tokS(data, LARGE, "base", "m8_latency")));
  assert.ok(close(fall.full, tokS(data, LARGE, "wkps", "m8_latency")));
  assert.ok(close(fall.steps.at(-1).after, fall.full));
  const [bestLabel, bestRate] = data.best[LARGE].m8_latency.fp8;
  assert.equal(fall.bestLabel, bestLabel);
  assert.ok(close(fall.bestGain, bestRate / fall.base));
  // in dollars a step that lowers tokens/s is a step that raises the cost
  const dollars = waterfall(data, LARGE, "m8_latency", "dollars");
  fall.steps.forEach((step, i) => assert.equal(step.worse, dollars.steps[i].worse));
  assert.equal(waterfall(data, LARGE, "no-such-workload"), null);
});

test("server rows: fastest first, controls only when asked for", () => {
  const rows = serverRows(data, LARGE, "m8_latency");
  assert.ok(rows.every((row, i) => i === 0 || rows[i - 1].gain >= row.gain));
  assert.ok(rows.every((row) => row.deployable && !row.label.endsWith("-r2")));
  assert.equal(rows[0].label, data.best[LARGE].m8_latency.any[0]);
  const withControls = serverRows(data, LARGE, "m8_latency", true);
  assert.ok(withControls.length > rows.length && withControls.some((row) => row.label === "sg"));
});

test("the page's recommendation map is Python's", () => {
  assert.deepEqual(MAP_USERS, data.map.users);
  assert.deepEqual(MAP_CONTEXTS, data.map.contexts);
  for (const [key, allowInt4] of [["fp8", false], ["any", true]]) {
    const grid = recommendationMap(data, LARGE, "informed", data.map.output_len, data.tokens_per_pass, allowInt4);
    assert.deepEqual(grid.map((row) => row.map((cell) => cell.label)), data.map.picks[key]);
  }
  const grid = recommendationMap(data, LARGE, "informed", data.map.output_len, data.tokens_per_pass, false);
  const { stars, checked } = mapAgreement(data, LARGE, grid);
  assert.equal(checked, 4);
  assert.ok(stars.length >= 1 && stars.length <= checked);
  assert.ok(data.map.picks.any.flat().every((label) => label in CANDIDATE_STACKS));
});

test("a preset in the calculator reproduces the prediction frozen before M8", () => {
  const letters = { w: { weights: "fp8" }, a: { weights: "int4" }, k: { fp8Kv: true }, p: { prefix: true } };
  let compared = 0;
  for (const row of data.predictions) {
    const on = row.label === "base" ? "" : row.label;
    if ([...on].some((letter) => !"wakps".includes(letter))) continue; // a control: not a calculator stack
    const workload = data.workloads[row.workload];
    const switches = Object.assign({}, ...[...on].map((letter) => letters[letter] ?? {}));
    const speculation = on.includes("s")
      ? { k: 3, tokens_per_pass: workload.tokens_per_pass, draft_bytes: data.draft_bytes[row.model] }
      : null;
    const stack = makeStack({ ...switches, speculation });
    const cfg = data.models[row.model];
    const load = makeLoad({ ...workload.load, kv_tokens: kvCapacity(cfg, stack, data.startup_memory[row.model]) });
    const got = predict(cfg, data.hardware, stack, data.calibrations.frozen, load);
    assert.ok(close(got.tok_s, row.tok_s, 1e-4), `${row.model} ${row.label} ${row.workload}: ${got.tok_s} vs ${row.tok_s}`);
    compared += 1;
  }
  assert.ok(compared >= 100, `only ${compared} predictions compared`);
});

test("prediction errors are grouped as in the figure", () => {
  const errors = predictionErrors(data);
  assert.equal(errors.rows.length, data.predictions.length);
  assert.ok(errors.groups["FP8 KV + speculation"] > errors.groups["no speculation"]);
  assert.ok(errors.within > 0 && errors.within < 1);
});

console.log(`${passed} tests passed`);
