/* Node parity check: the browser classifier must produce identical verdicts
   and probabilities to the Python CLI on real held-out test messages. */

import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { setModel, classify } from "./spamcheck.js";

setModel(JSON.parse(readFileSync("model.json", "utf-8")));

const rows = readFileSync("data/test.jsonl", "utf-8")
  .split("\n").filter(Boolean).map(l => JSON.parse(l));

// a deterministic slice: first 3 rows, first 5 spam rows, every 150th row
const spamRows = rows.filter(r => r.label === "spam");
const picks = [...rows.slice(0, 3), ...spamRows.slice(0, 5),
               ...rows.filter((_, i) => i % 150 === 0)].slice(0, 12);
writeFileSync("parity-samples.json", JSON.stringify(picks), "utf-8");

const py = `
import json, sys
sys.path.insert(0, ".")
import spamcheck as sc
model = sc.load_model()
rows = json.load(open("parity-samples.json", encoding="utf-8"))
out = [sc.classify(model, r["text"])[:2] for r in rows]
print(json.dumps([[v, p] for v, p in out]))
`;

const pyOut = execFileSync("python", ["-X", "utf8", "-c", py], { encoding: "utf-8" });
const pyResults = JSON.parse(pyOut.trim());

let fail = 0;
pyResults.forEach(([verdict, prob], i) => {
  const js = classify(picks[i].text);
  const sameVerdict = js.verdict === verdict;
  const sameProb = Math.abs(js.prob - prob) < 1e-9;
  const label = picks[i].label;
  const agreesWithTruth = verdict === label;
  console.log(`msg ${String(i).padStart(2)}: py=${verdict}/${prob.toFixed(6)} js=${js.verdict}/${js.prob.toFixed(6)} ` +
    `${sameVerdict ? "verdict-ok" : "VERDICT-DRIFT"} ${sameProb ? "prob-ok" : "PROB-DRIFT"} truth=${label}`);
  if (!sameVerdict || !sameProb) fail += 1;
});
console.log(fail ? `FAIL: ${fail} drifts` : "parity: JS === Python on all samples");
process.exit(fail ? 1 : 0);
