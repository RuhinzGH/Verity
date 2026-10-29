"""Evaluation metrics: retrieval, classification and calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


# ------------------------------------------------------------------ retrieval
def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold)


def hit_at(ranked: list[str], gold: set[str], k: int) -> float:
    return float(bool(set(ranked[:k]) & gold))


def mrr(ranked: list[str], gold: set[str], k: int = 10) -> float:
    for i, pid in enumerate(ranked[:k]):
        if pid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, pid in enumerate(ranked[:k]) if pid in gold)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal


# ------------------------------------------------------------------ classification
def classification(y_true, y_pred, labels) -> dict:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(n[i])} for i, l in enumerate(labels)},
    }


def bootstrap_ci(y_true, y_pred, fn, n: int = 1000, seed: int = 42, groups=None) -> tuple[float, float]:
    """95% percentile bootstrap interval for a metric fn(y_true, y_pred).

    With `groups` (e.g. claim ids), whole groups are resampled together: Climate-FEVER has
    five evidence pairs per claim, and resampling pairs independently would ignore that
    they are not independent, making the interval too narrow."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if groups is None:
        members = [np.array([i]) for i in range(len(y_true))]
    else:
        g = np.asarray(groups)
        members = [np.flatnonzero(g == u) for u in dict.fromkeys(g.tolist())]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(members), len(members))
        i = np.concatenate([members[j] for j in pick])
        vals.append(fn(y_true[i], y_pred[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def macro_f1(labels):
    return lambda t, p: f1_score(t, p, labels=labels, average="macro", zero_division=0)


# ------------------------------------------------------------------ calibration
def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-label confidence (equal-width bins)."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def reliability(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "confidence": float(conf[m].mean()), "accuracy": float((pred[m] == y[m]).mean())})
    return out
