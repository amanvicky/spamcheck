#!/usr/bin/env python3
"""spamcheck — classify a message with the trained model.json.

Zero dependencies at inference time: the model is one JSON file of log
probabilities, and this script does the math the trainer exported. The
JavaScript demo in demo/ runs the identical computation in your browser.

    python spamcheck.py "URGENT! You have won a FREE prize, call now"
    -> SPAM  99.6%

"""

import json
import math
import re
import sys
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_model(path=None):
    path = path or Path(__file__).resolve().parent / "model.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log_scores(model, text):
    """{class: log P(class | text)} — unnormalized log posterior."""
    tokens = tokenize(text)
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    out = {}
    for c, params in model["classes"].items():
        lp = params["prior_log"]
        table = params["log_probs"]
        floor = math.log(1.0 / 10_000_000)   # unseen tokens: tiny, equal for both classes
        for t, n in counts.items():
            lp += n * table.get(t, floor)
        out[c] = lp
    return out, tokens


def classify(model, text):
    scores, tokens = log_scores(model, text)
    # log-sum-exp -> probabilities
    top = max(scores.values())
    exp = {c: math.exp(v - top) for c, v in scores.items()}
    total = sum(exp.values())
    probs = {c: e / total for c, e in exp.items()}
    verdict = max(probs, key=probs.get)
    return verdict, probs[verdict], tokens, scores


def top_pushers(model, tokens, scores):
    """The tokens that pushed the verdict, biggest absolute contribution first."""
    per = {}
    for t in set(tokens):
        spam_lp = model["classes"]["spam"]["log_probs"].get(t, model["classes"]["spam"]["log_probs"].get("__floor__", -16.12))
        ham_lp = model["classes"]["ham"]["log_probs"].get(t, -16.12)
        per[t] = spam_lp - ham_lp            # positive = pushes SPAM
    return sorted(per.items(), key=lambda kv: -abs(kv[1]))[:8]


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python spamcheck.py "message text"', file=sys.stderr)
        return 1
    model = load_model()
    text = " ".join(argv)
    verdict, prob, tokens, scores = classify(model, text)
    print(f"{verdict.upper()}  {prob * 100:.1f}%")
    for t, push in top_pushers(model, tokens, scores)[:6]:
        side = "spam" if push > 0 else "ham"
        print(f"  {t!r:20} pushes {side}  (Δlog {push:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
