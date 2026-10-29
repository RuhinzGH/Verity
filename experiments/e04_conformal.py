"""Experiment 4: can the uncertainty labels be trusted?

For the zero-shot and the fine-tuned verifier:
  1. fit a temperature on the SciFact calibration claims (20% of train, never trained on)
  2. fit the conformal threshold at alpha = 0.05, 0.10, 0.20 on the same claims
  3. test on SciFact dev (same distribution as calibration) and on Climate-FEVER
     (a different distribution: the guarantee is not expected to hold there, and the
     experiment measures by how much it breaks)

Also records ECE before and after temperature scaling, reliability bins, and a
risk-coverage curve (accuracy when the verifier answers only its most confident x%).

The fine-tuned model at alpha = 0.10 is written to the "deployed" block, which the
pipeline and the app read.
"""
import json

import numpy as np

from verity import conformal as C, metrics as M, pairs as P
from verity.config import FINETUNED_NLI, L2I, RESULTS
from verity.verify import cached_logits, softmax

MODELS = {"deberta-fever": "deberta-fever", "verity-finetuned": str(FINETUNED_NLI)}
ALPHAS = [0.05, 0.10, 0.20]


def labels(pairs):
    return np.array([L2I[p["label"]] for p in pairs])


def risk_coverage(probs, y, points=20):
    conf, ok = probs.max(1), probs.argmax(1) == y
    order = np.argsort(-conf)
    out = []
    for frac in np.linspace(1 / points, 1, points):
        k = max(1, int(round(frac * len(y))))
        out.append({"coverage": float(frac), "accuracy": float(ok[order[:k]].mean())})
    return out


def main():
    _, cal = P.split_train_calibration(P.scifact_pairs("train"))
    tests = {"scifact_dev": P.scifact_pairs("dev"), "climate_fever": P.climate_pairs()}
    out = {"n_calibration": len(cal), "models": {}}
    for name, path in MODELS.items():
        zc = cached_logits(path, [(p["evidence"], p["claim"]) for p in cal], "scifact_cal")
        yc = labels(cal)
        t = C.fit_temperature(zc, yc)
        pc = softmax(zc, t)
        res = {"temperature": t, "calibration_ece_before": M.ece(softmax(zc), yc),
               "calibration_ece_after": M.ece(pc, yc), "tests": {}}
        for tname, pairs in tests.items():
            z = cached_logits(path, [(p["evidence"], p["claim"]) for p in pairs], tname)
            y = labels(pairs)
            p_raw, p_t = softmax(z), softmax(z, t)
            r = {"n": len(y), "ece_before": M.ece(p_raw, y), "ece_after": M.ece(p_t, y),
                 "reliability_after": M.reliability(p_t, y), "risk_coverage": risk_coverage(p_t, y),
                 "conformal": {}}
            for a in ALPHAS:
                q = C.calibrate(pc, yc, a)
                r["conformal"][str(a)] = {"qhat": q, "target_coverage": 1 - a, **C.summary(p_t, y, q)}
            res["tests"][tname] = r
            c10 = r["conformal"]["0.1"]
            print(f"{name:17s} {tname:14s} T={t:.2f}  ECE {r['ece_before']:.3f} -> {r['ece_after']:.3f}  "
                  f"| alpha .10: coverage {c10['coverage']:.3f}  set size {c10['mean_set_size']:.2f}  "
                  f"confident {c10['share_confident']:.2f}  acc|confident {c10['accuracy_when_confident']}",
                  flush=True)
        out["models"][name] = res
    ft = out["models"]["verity-finetuned"]
    out["deployed"] = {"model": "verity-finetuned", "alpha": 0.1, "temperature": ft["temperature"],
                       "qhat": ft["tests"]["scifact_dev"]["conformal"]["0.1"]["qhat"]}
    (RESULTS / "e04_conformal.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
