import math

import numpy as np
import pytest

from verity import metrics as M


RANKED = ["a", "b", "c", "d", "e"]


def test_recall_hit():
    gold = {"b", "e", "z"}
    assert M.recall_at(RANKED, gold, 1) == 0
    assert M.recall_at(RANKED, gold, 2) == pytest.approx(1 / 3)
    assert M.recall_at(RANKED, gold, 5) == pytest.approx(2 / 3)
    assert M.hit_at(RANKED, gold, 1) == 0.0
    assert M.hit_at(RANKED, gold, 2) == 1.0
    assert M.recall_at(RANKED, {"a"}, 1) == 1.0


def test_mrr():
    assert M.mrr(RANKED, {"c"}) == pytest.approx(1 / 3)
    assert M.mrr(RANKED, {"c", "a"}) == 1.0
    assert M.mrr(RANKED, {"e"}, k=4) == 0.0
    assert M.mrr(RANKED, {"zz"}) == 0.0


def test_ndcg():
    assert M.ndcg(RANKED, {"a"}) == pytest.approx(1.0)
    assert M.ndcg(RANKED, {"a", "b"}) == pytest.approx(1.0)
    # gold at rank 2 only: (1/log2 3) / 1
    assert M.ndcg(RANKED, {"b"}) == pytest.approx(1 / math.log2(3))
    # gold at ranks 2 and 4: (1/log2 3 + 1/log2 5) / (1 + 1/log2 3)
    exp = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert M.ndcg(RANKED, {"b", "d"}) == pytest.approx(exp)
    assert M.ndcg(RANKED, {"zz"}) == 0.0
    # more gold than k: ideal is capped at k
    assert M.ndcg(RANKED, set(RANKED) | {"x", "y"}, k=3) == pytest.approx(1.0)


def test_classification():
    labels = ["SUPPORTED", "REFUTED", "NEI"]
    y = ["SUPPORTED", "SUPPORTED", "REFUTED", "NEI"]
    p = ["SUPPORTED", "REFUTED", "REFUTED", "SUPPORTED"]
    r = M.classification(y, p, labels)
    assert r["n"] == 4
    assert r["accuracy"] == pytest.approx(0.5)
    pc = r["per_class"]
    assert pc["SUPPORTED"] == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2})
    assert pc["REFUTED"]["precision"] == pytest.approx(0.5)
    assert pc["REFUTED"]["recall"] == pytest.approx(1.0)
    assert pc["REFUTED"]["f1"] == pytest.approx(2 / 3)
    assert pc["NEI"] == pytest.approx({"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1})
    assert r["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 0) / 3)


def test_ece_perfectly_calibrated_is_zero():
    # every prediction has confidence 0.75 and 3 of 4 are right
    probs = np.array([[0.75, 0.25]] * 4)
    y = np.array([0, 0, 0, 1])
    assert M.ece(probs, y) == pytest.approx(0.0, abs=1e-12)
    # two bins, each calibrated on its own
    probs = np.array([[0.75, 0.25]] * 4 + [[0.05, 0.95]] * 20)
    y = np.array([0, 0, 0, 1] + [1] * 19 + [0])
    assert M.ece(probs, y) == pytest.approx(0.0, abs=1e-12)


def test_ece_hand_computed():
    # bin (0.8,0.9]: 2 items conf 0.9, 1 correct -> |0.5-0.9| = 0.4, weight 2/4
    # bin (0.5,0.6]: 2 items conf 0.6, 2 correct -> |1.0-0.6| = 0.4, weight 2/4
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.4, 0.6]])
    y = np.array([0, 1, 1, 1])
    assert M.ece(probs, y) == pytest.approx(0.4)
    # mixed: bin (0.8,0.9] conf 0.9 acc 1 -> 0.1 * 1/2; bin (0.6,0.7] conf 0.7 acc 0 -> 0.7 * 1/2
    probs = np.array([[0.9, 0.1], [0.3, 0.7]])
    y = np.array([0, 0])
    assert M.ece(probs, y) == pytest.approx(0.5 * 0.1 + 0.5 * 0.7)
    # a single overconfident wrong prediction
    assert M.ece(np.array([[1.0, 0.0]]), np.array([1])) == pytest.approx(1.0)
    rel = M.reliability(np.array([[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.4, 0.6]]), np.array([0, 1, 1, 1]))
    assert [b["n"] for b in rel] == [2, 2]
    assert [b["accuracy"] for b in rel] == [1.0, 0.5]


def test_bootstrap_ci_ordered_and_contains_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    p = np.where(rng.random(200) < 0.7, y, (y + 1) % 3)
    acc = lambda t, q: float((t == q).mean())
    lo, hi = M.bootstrap_ci(y, p, acc, n=300)
    assert lo <= acc(y, p) <= hi
    assert 0 <= lo < hi <= 1
    assert M.bootstrap_ci(y, p, acc, n=300) == (lo, hi)        # seeded, deterministic
    lo2, hi2 = M.bootstrap_ci(y, p, M.macro_f1([0, 1, 2]), n=200)
    assert lo2 <= hi2
