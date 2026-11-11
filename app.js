/* UI for spamcheck: verdict badge, probability bar, contribution chips.
   The classifier lives in spamcheck.js (pure, Node-testable). */

import { setModel, classify, topPushers, modelInfo } from "./spamcheck.js";

const SAMPLES = {
  spam: "URGENT! You have won a FREE entry in our £1000 prize draw. Call now to claim your cash prize!",
  ham: "hey are we still meeting at the library at 5 to finish the assignment?",
  tricky: "Your parcel could not be delivered. Reschedule delivery by clicking the link and paying £1.45 redelivery fee.",
};

function render(text) {
  const { verdict, prob, probs } = classify(text);
  const badge = document.getElementById("verdict");
  badge.textContent = verdict.toUpperCase();
  badge.className = "verdict " + verdict;

  const bar = document.getElementById("bar");
  const pct = Math.round((probs.spam) * 1000) / 10;
  bar.style.width = pct + "%";
  bar.className = "bar " + (verdict === "spam" ? "spambar" : "hambar");

  document.getElementById("prob").textContent =
    `${pct.toFixed(1)}% spam · ${(100 - pct).toFixed(1)}% ham`;

  const box = document.getElementById("pushers");
  box.innerHTML = "";
  const pushers = topPushers(text);
  if (!pushers.length) {
    box.innerHTML = '<span class="dim">no recognized tokens — type a message</span>';
    return;
  }
  pushers.forEach(({ token, delta, pushes }) => {
    const chip = document.createElement("span");
    chip.className = "chip " + pushes;
    chip.textContent = `${token} ${delta > 0 ? "+" : ""}${delta.toFixed(2)}`;
    chip.title = `Δ log P(spam)/P(ham) = ${delta.toFixed(3)} — pushes ${pushes}`;
    box.appendChild(chip);
  });
}

function renderModelInfo() {
  const m = modelInfo();
  document.getElementById("modelInfo").innerHTML =
    `trained on <b>${m.examples.toLocaleString()}</b> real SMS messages<br>` +
    `held-out accuracy <b>${(m.accuracy * 100).toFixed(2)}%</b><br>` +
    `vocabulary <b>${m.vocab.toLocaleString()}</b> tokens<br>` +
    `trained <b>${m.trained_at.slice(0, 10)}</b>`;
}

const input = document.getElementById("input");
input.addEventListener("input", () => render(input.value));
document.querySelectorAll("button[data-sample]").forEach(btn => {
  btn.addEventListener("click", () => {
    input.value = SAMPLES[btn.dataset.sample];
    render(input.value);
  });
});

(async function start() {
  try {
    const res = await fetch("model.json");
    setModel(await res.json());
    renderModelInfo();
    render(input.value);
  } catch (e) {
    document.getElementById("verdict").textContent = "model failed to load";
    document.getElementById("verdict").className = "verdict spam";
  }
})();
