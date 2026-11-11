"""Tests for spamcheck: the shipped model, real eval on the held-out split,
and the CLI surface. The model artifact itself is what's under test."""

import json
import math
from pathlib import Path

import pytest

import spamcheck as sc

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def model():
    return sc.load_model()


def test_model_artifact_shape(model):
    assert model["model"] == "multinomial-nb"
    assert model["dataset"]["examples"] == 5574
    assert model["dataset"]["train"] + model["dataset"]["test"] == 5574
    assert model["vocab_size"] > 5000
    for c in ("spam", "ham"):
        assert len(model["classes"][c]["log_probs"]) == model["vocab_size"]


def test_obvious_spam_scores_spam(model):
    verdict, prob, tokens, scores = sc.classify(
        model, "URGENT! You have won a FREE prize. Call 09061234567 now to claim your cash!")
    assert verdict == "spam"
    assert prob > 0.9


def test_obvious_ham_scores_ham(model):
    verdict, prob, _, _ = sc.classify(
        model, "hey are we still meeting at the library at 5 to finish the assignment?")
    assert verdict == "ham"
    assert prob > 0.9


def test_tokenize_is_lowercase_words():
    assert sc.tokenize("URGENT!! Free entry — call 0906 1234567") == \
        ["urgent", "free", "entry", "call", "0906", "1234567"]


def test_eval_on_shipped_test_split(model):
    """The headline metric, recomputed from the shipped artifacts every CI run."""
    lines = (ROOT / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines]
    assert len(rows) == model["dataset"]["test"]
    correct = sum(sc.classify(model, r["text"])[0] == r["label"] for r in rows)
    accuracy = correct / len(rows)
    assert accuracy >= 0.95, accuracy
    assert accuracy == pytest.approx(model["metrics"]["accuracy"], abs=0.001)


def test_probabilities_are_normalized(model):
    verdict, prob, _, scores = sc.classify(model, "free prize urgent win claim cash now")
    assert verdict == "spam" and 0.99 <= prob <= 1.0
    top = max(scores.values())
    total = sum(math.exp(v - top) for v in scores.values())
    assert sum(math.exp(v - top) / total for v in scores.values()) == pytest.approx(1.0)


def test_top_pushers_are_explainable(model):
    _, _, tokens, scores = sc.classify(model, "URGENT! You have won a FREE prize, claim now")
    pushers = sc.top_pushers(model, tokens, scores)
    assert any(t == "urgent" and push > 0 for t, push in pushers)


def test_main_prints_verdict(capsys):
    assert sc.main(["URGENT! Free prize, call now to claim"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("SPAM") and "pushes spam" in out


def test_main_no_args_errors():
    assert sc.main([]) == 1
