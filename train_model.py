#!/usr/bin/env python3
"""Train the spam classifier on the UCI SMS Spam Collection and export it.

The model is a hand-rolled Multinomial Naive Bayes over word counts — written
out by hand on purpose, so the exported model.json is exactly the training
math and a JavaScript port can match it token for token:

    score(class) = log_prior(class) + sum over tokens: count * log_p(class, token)
    probability = softmax over class scores (log-sum-exp for stability)

Outputs:
  data/train.jsonl, data/test.jsonl  — stratified 80/20 split, seed 7
  model.json                         — priors + per-token log probs + metrics
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9']+")
ALPHA = 1.0
SEED = 7

SOURCE = {
    "name": "SMS Spam Collection v.1 (UCI Machine Learning Repository, id 228)",
    "url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
    "citation": "Almeida, T.A., Hidalgo, J.M.G., Ling, A.H.N. (2011). A dataset for SMS Spam Collection. Journal of Information and Data Management 2(1).",
    "license_note": "Free for research and education with the citation above.",
}


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_raw(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        out.append((label.strip(), text.strip()))
    return out


def stratified_split(rows, ratio=0.8, seed=SEED):
    by_class = {}
    for label, text in rows:
        by_class.setdefault(label, []).append(text)
    train, test = [], []
    for label, texts in sorted(by_class.items()):
        texts = sorted(texts)
        rng = random.Random(seed)
        rng.shuffle(texts)
        cut = round(len(texts) * ratio)
        train += [(label, t) for t in texts[:cut]]
        test += [(label, t) for t in texts[cut:]]
    return train, test


def train(rows):
    counts = {c: Counter() for c in ("spam", "ham")}
    docs = Counter()
    for label, text in rows:
        counts[label].update(tokenize(text))
        docs[label] += 1
    total_docs = sum(docs.values())
    vocab = sorted(set(t for c in counts for t in counts[c]))
    model = {"classes": {}}
    for c in ("spam", "ham"):
        total_tokens = sum(counts[c].values())
        prior = docs[c] / total_docs
        denom = total_tokens + ALPHA * len(vocab)
        model["classes"][c] = {
            "prior_log": __import__("math").log(prior),
            "log_probs": {t: __import__("math").log((counts[c][t] + ALPHA) / denom) for t in vocab},
        }
    return model, vocab, docs


def main():
    root = Path(__file__).resolve().parent
    raw = load_raw(root / "data" / "raw" / "SMSSpamCollection")
    train_rows, test_rows = stratified_split(raw)

    (root / "data" / "train.jsonl").write_text(
        "".join(json.dumps({"label": l, "text": t}, ensure_ascii=False) + "\n" for l, t in train_rows),
        encoding="utf-8")
    (root / "data" / "test.jsonl").write_text(
        "".join(json.dumps({"label": l, "text": t}, ensure_ascii=False) + "\n" for l, t in test_rows),
        encoding="utf-8")

    model, vocab, docs = train(train_rows)

    # evaluate on the held-out split
    scores = json.loads(json.dumps(model))
    correct = 0
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    sys.path.insert(0, str(root))
    from spamcheck import classify  # the same inference the CLI ships
    for label, text in test_rows:
        predicted = classify(scores, text)[0]
        correct += predicted == label
        if label == "spam":
            confusion["tp" if predicted == "spam" else "fn"] += 1
        else:
            confusion["tn" if predicted == "ham" else "fp"] += 1
    accuracy = correct / len(test_rows)
    tp, fp, tn, fn = (confusion[k] for k in ("tp", "fp", "tn", "fn"))
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_spam": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall_spam": round(tp / (tp + fn), 4),
        "f1_spam": round(2 * (tp / (tp + fp)) * (tp / (tp + fn)) /
                         ((tp / (tp + fp)) + (tp / (tp + fn))), 4) if tp + fp else None,
        "confusion": confusion, "test_examples": len(test_rows),
    }

    doc = {
        "model": "multinomial-nb",
        "version": 1,
        "trained_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "tokenizer": "lowercase, regex [a-z0-9']+",
        "smoothing_alpha": ALPHA,
        "dataset": {**SOURCE, "examples": len(raw), "train": len(train_rows), "test": len(test_rows)},
        "metrics": metrics,
        "vocab_size": len(vocab),
        "classes": model["classes"],
    }
    (root / "model.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"train {len(train_rows)} / test {len(test_rows)} | vocab {len(vocab):,} | "
          f"accuracy {accuracy:.4f} | P={metrics['precision_spam']} R={metrics['recall_spam']} "
          f"F1={metrics['f1_spam']}")


if __name__ == "__main__":
    main()
