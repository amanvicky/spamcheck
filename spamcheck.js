/* Multinomial Naive Bayes inference — mirrors spamcheck.py exactly.
   Pure module: no DOM, so Node can run it against the Python implementation.
   score(class) = log_prior + sum(count * log_p); probability = softmax. */

let MODEL = null;

export function setModel(doc) {
  MODEL = doc;
}

export function tokenize(text) {
  return text.toLowerCase().match(/[a-z0-9']+/g) || [];
}

function logScores(text) {
  const tokens = tokenize(text);
  const counts = {};
  for (const t of tokens) counts[t] = (counts[t] || 0) + 1;
  const out = {};
  for (const [c, params] of Object.entries(MODEL.classes)) {
    let lp = params.prior_log;
    const table = params.log_probs;
    const floor = Math.log(1.0 / 10000000);
    for (const [t, n] of Object.entries(counts)) {
      lp += n * (table[t] !== undefined ? table[t] : floor);
    }
    out[c] = lp;
  }
  return { scores: out, tokens };
}

export function classify(text) {
  const { scores, tokens } = logScores(text);
  const top = Math.max(...Object.values(scores));
  const exp = {};
  let total = 0;
  for (const [c, v] of Object.entries(scores)) {
    exp[c] = Math.exp(v - top);
    total += exp[c];
  }
  const probs = {};
  for (const [c, e] of Object.entries(exp)) probs[c] = e / total;
  const verdict = Object.keys(probs).reduce((a, b) => (probs[a] > probs[b] ? a : b));
  return { verdict, prob: probs[verdict], probs, tokens };
}

export function topPushers(text, limit = 6) {
  const { tokens } = logScores(text);
  const per = {};
  const floor = Math.log(1.0 / 10000000);
  const spamTable = MODEL.classes.spam.log_probs;
  const hamTable = MODEL.classes.ham.log_probs;
  for (const t of new Set(tokens)) {
    per[t] = (spamTable[t] !== undefined ? spamTable[t] : floor) -
             (hamTable[t] !== undefined ? hamTable[t] : floor);
  }
  return Object.entries(per)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, limit)
    .map(([token, delta]) => ({ token, delta, pushes: delta > 0 ? "spam" : "ham" }));
}

export function modelInfo() {
  return {
    examples: MODEL.dataset.examples,
    accuracy: MODEL.metrics.accuracy,
    vocab: MODEL.vocab_size,
    trained_at: MODEL.trained_at,
  };
}
