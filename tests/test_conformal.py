import numpy as np
import pytest

from verity import conformal
from verity.config import LABELS


def draw(rng, n):
    """3-class Dirichlet probabilities with labels sampled from them (perfectly calibrated)."""
    probs = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    y = (rng.random(n)[:, None] > probs.cumsum(1)).sum(1)
    return probs, np.minimum(y, 2)


def test_coverage_guarantee_holds_on_fresh_data():
    rng = np.random.default_rng(0)
    cal_p, cal_y = draw(rng, 2000)
    qhat = conformal.calibrate(cal_p, cal_y, alpha=0.1)
    assert 0 < qhat <= 1
    test_p, test_y = draw(rng, 50_000)
    cov = conformal.coverage(conformal.sets(test_p, qhat), test_y)
    assert cov >= 0.88
    assert cov <= 0.95          # not trivially full sets either
    s = conformal.summary(test_p, test_y, qhat)
    assert s["coverage"] == pytest.approx(cov)
    assert 1 < s["mean_set_size"] < 3


@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.3])
def test_coverage_other_alphas(alpha):
    rng = np.random.default_rng(1)
    cal_p, cal_y = draw(rng, 3000)
    qhat = conformal.calibrate(cal_p, cal_y, alpha=alpha)
    tp, ty = draw(rng, 30_000)
    assert conformal.coverage(conformal.sets(tp, qhat), ty) >= 1 - alpha - 0.02


def test_calibrate_small_n_returns_max_score():
    # n=5, alpha=0.1 -> level = ceil(6*0.9)/5 > 1 -> clipped to 1 -> the largest score
    probs = np.array([[0.9, 0.05, 0.05]] * 5)
    y = np.array([0, 0, 0, 1, 2])
    assert conformal.calibrate(probs, y, 0.1) == pytest.approx(0.95)


def test_sets_and_verdicts():
    probs = np.array([
        [0.90, 0.05, 0.05],   # confident SUPPORTED
        [0.10, 0.85, 0.05],   # confident REFUTED
        [0.45, 0.45, 0.10],   # two labels -> UNCERTAIN
        [0.34, 0.33, 0.33],   # nothing reaches 0.4 -> empty -> UNCERTAIN
    ])
    qhat = 0.6        # a label is kept if p >= 0.4
    m = conformal.sets(probs, qhat)
    assert m.shape == (4, 3)
    assert m.sum(1).tolist() == [1, 1, 2, 0]
    v = conformal.verdicts(probs, qhat)
    assert [x["verdict"] for x in v] == ["SUPPORTED", "REFUTED", "UNCERTAIN", "UNCERTAIN"]
    assert v[0]["set"] == ["SUPPORTED"]
    assert v[2]["set"] == ["SUPPORTED", "REFUTED"]
    assert v[3]["set"] == []
    for x, p in zip(v, probs):
        assert list(x["probs"]) == LABELS
        assert list(x["probs"].values()) == pytest.approx(p.tolist())
    v = conformal.verdicts(probs[3:], 0.9)
    assert v[0]["verdict"] == "UNCERTAIN" and v[0]["set"] == LABELS
    v = conformal.verdicts(np.array([[0.1, 0.1, 0.8]]), 0.5)
    assert v[0]["verdict"] == "NEI" and v[0]["set"] == ["NEI"]


def test_summary_fields():
    probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.85, 0.05], [0.45, 0.45, 0.10]])
    y = np.array([0, 2, 1])
    s = conformal.summary(probs, y, 0.6)
    assert s["coverage"] == pytest.approx(2 / 3)
    assert s["share_confident"] == pytest.approx(2 / 3)
    assert s["accuracy_when_confident"] == pytest.approx(0.5)
    assert s["share_empty"] == 0.0


@pytest.mark.parametrize("T", [0.5, 2.0, 4.0])
def test_fit_temperature_recovers_known_temperature(T):
    rng = np.random.default_rng(2)
    n = 20_000
    true_logits = rng.normal(0, 2.5, size=(n, 3))
    p = np.exp(true_logits - true_logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > p.cumsum(1)).sum(1).clip(0, 2)
    # the model reports true_logits * T, so dividing by the fitted temperature should undo it
    t = conformal.fit_temperature(true_logits * T, y)
    assert t == pytest.approx(T, rel=0.1)
