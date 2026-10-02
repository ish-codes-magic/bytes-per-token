// Does the browser's model (site/model.js) reproduce the Python model's numbers?
//   node site/tests/parity.mjs
// Exits 1 and prints the first differences if not. The test suite runs this where Node is installed.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { runChecks } from "../model.js";

const here = dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(readFileSync(join(here, "..", "data", "dashboard.json"), "utf-8"));
const failures = runChecks(data);
if (failures.length > 0) {
  for (const f of failures.slice(0, 10)) {
    const stack = JSON.stringify(f.check.stack);
    console.error(`${f.check.model} ${f.check.calibration} ${stack} users=${f.check.load.users}`);
    console.error(`  ${f.key}: Python ${f.want}, JavaScript ${f.got} (relative error ${f.error.toExponential(2)})`);
  }
  console.error(`${failures.length} values differ over ${data.checks.length} checks`);
  process.exit(1);
}
console.log(`the JavaScript model matches the Python model on all ${data.checks.length} checks`);
