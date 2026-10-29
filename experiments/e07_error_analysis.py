"""Experiment 7: where does the verifier fail?

Tags every SciFact dev pair and Climate-FEVER (evidence, claim) pair by simple,
rule-based traits of the claim and compares the fine-tuned verifier's accuracy (oracle
evidence, experiment 2) on pairs with and without each trait. The rules are
deliberately crude and listed here so the reader can judge them:

  number      the claim contains a digit, a percentage or a number word
  negation    not, no, never, without, fails, lack(s), cannot, n't
  comparison  more, less, higher, lower, increase(s), decrease(s), reduce(s), than
  hedged      may, might, can, could, associated, linked, likely
  long        more than 20 words

It also lists the most confident mistakes (the errors that matter most, because the
system would show them as a firm verdict) and the confusion matrix.

Reads results/e02_predictions.json; writes results/e07_error_analysis.json.
"""
import json
import re

import numpy as np

from verity.config import LABELS, RESULTS

TRAITS = {
    "number": r"\d|%|\b(one|two|three|four|five|ten|hundred|thousand|million|half|twice|double)\b",
    "negation": r"\b(not|no|never|without|fails?|lacks?|cannot)\b|n't",
    "comparison": r"\b(more|less|higher|lower|greater|fewer|increase[sd]?|decrease[sd]?|reduce[sd]?|than)\b",
    "hedged": r"\b(may|might|can|could|associated|linked|likely)\b",
    "long": None,
}
MODEL = "verity-finetuned"


def has(trait, text):
    if trait == "long":
        return len(text.split()) > 20
    return re.search(TRAITS[trait], text, re.I) is not None


def analyse(block):
    pairs = block["pairs"]
    probs = np.array(block[MODEL])
    y = np.array([LABELS.index(p["label"]) for p in pairs])
    pred = probs.argmax(1)
    ok = pred == y
    out = {"n": len(y), "accuracy": float(ok.mean()), "traits": {},
           "confusion": {LABELS[t]: {LABELS[q]: int(((y == t) & (pred == q)).sum()) for q in range(3)}
                         for t in range(3)}}
    for t in TRAITS:
        m = np.array([has(t, p["claim"]) for p in pairs])
        if m.sum() >= 10 and (~m).sum() >= 10:
            out["traits"][t] = {"n_with": int(m.sum()), "acc_with": float(ok[m].mean()),
                                "acc_without": float(ok[~m].mean()),
                                "gap": float(ok[m].mean() - ok[~m].mean())}
    conf = probs.max(1)
    sup = LABELS.index("SUPPORTED")
    wrong_sup = (y == sup) & ~ok
    nei_on_sup = wrong_sup & (pred == LABELS.index("NEI"))
    out["mistakes_on_supported"] = {"total": int(wrong_sup.sum()), "said_nei": int(nei_on_sup.sum()),
                                    "said_nei_confident_over_0.9": int((nei_on_sup & (conf > 0.9)).sum()),
                                    "median_confidence_of_nei_mistakes": float(np.median(conf[nei_on_sup])) if nei_on_sup.any() else None}
    worst = [i for i in np.argsort(-conf) if not ok[i]][:8]
    out["confident_mistakes"] = [{"claim": pairs[i]["claim"], "evidence": pairs[i]["evidence"][:400],
                                  "gold": pairs[i]["label"], "predicted": LABELS[pred[i]],
                                  "confidence": float(conf[i])} for i in worst]
    return out


def main():
    preds = json.loads((RESULTS / "e02_predictions.json").read_text())
    res = {k: analyse(preds[k]) for k in ["scifact_dev", "climate_fever"] if MODEL in preds.get(k, {})}
    for k, r in res.items():
        print(f"{k}: accuracy {r['accuracy']:.3f} (n={r['n']})  mistakes on supported: {r['mistakes_on_supported']}")
        for t, v in r["traits"].items():
            print(f"   {t:11s} n={v['n_with']:5d}  with {v['acc_with']:.3f}  without {v['acc_without']:.3f}  gap {v['gap']:+.3f}")
    (RESULTS / "e07_error_analysis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
