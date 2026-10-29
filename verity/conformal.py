"""Uncertainty labels with a coverage guarantee (split conformal prediction).

A verifier outputs probabilities. On a held-out calibration set we find the threshold
q such that, for a new claim drawn from the same distribution, the *set* of labels
{y : p(y) >= 1 - q} contains the true label with probability at least 1 - alpha.

Verity reports:
  one label in the set   -> that label (a confident verdict)
  two or three labels    -> UNCERTAIN, listing the labels it cannot rule out
  no label               -> UNCERTAIN (nothing is plausible enough)

The guarantee is about the long-run rate over many claims, and only holds when test
claims resemble the calibration claims (exchangeability). Experiment 4 tests what
happens when they do not.

Temperature scaling (one parameter, fit on the same calibration set) is applied first
so the probabilities themselves are better calibrated; it does not change the ranking.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .config import LABELS


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(t):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def calibrate(probs: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> float:
    """Threshold on the non-conformity score 1 - p(true label) (LAC)."""
    n = len(y)
    scores = 1.0 - probs[np.arange(n), y]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean (n, 3) membership of each label in the prediction set."""
    return (1.0 - probs) <= qhat


def coverage(member: np.ndarray, y: np.ndarray) -> float:
    return float(member[np.arange(len(y)), y].mean())


def verdicts(probs: np.ndarray, qhat: float) -> list[dict]:
    out = []
    for p, m in zip(probs, sets(probs, qhat)):
        labs = [LABELS[i] for i in np.flatnonzero(m)]
        out.append({"verdict": labs[0] if len(labs) == 1 else "UNCERTAIN",
                    "set": labs, "probs": {l: float(v) for l, v in zip(LABELS, p)}})
    return out


def summary(probs: np.ndarray, y: np.ndarray, qhat: float) -> dict:
    m = sets(probs, qhat)
    size = m.sum(1)
    single = size == 1
    pred = probs.argmax(1)
    return {
        "coverage": coverage(m, y),
        "mean_set_size": float(size.mean()),
        "share_confident": float(single.mean()),
        "accuracy_when_confident": float((m[single].argmax(1) == y[single]).mean()) if single.any() else None,
        "accuracy_all": float((pred == y).mean()),
        "share_empty": float((size == 0).mean()),
    }
