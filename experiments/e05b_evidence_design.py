"""Experiment 5b: how should the end-to-end pipeline cut evidence out of an abstract?

Motivation (first version, before the audit): experiment 5 found the fine-tuned verifier
no better than the zero-shot one once it had to find its own evidence. A suspected cause
was the format of its training evidence. The audit later found the real cause (NEI
evidence was always three sentences long); with length-matched pairs the fine-tuned
verifier is ahead end to end, and this experiment is kept as the design choice.

Three ways to present an abstract to the verifier:
  top3-concat   the 3 most similar sentences joined (experiment 5's design)
  per-sentence  each of the 5 most similar sentences scored on its own; strongest wins
  windows       every 2-sentence window of the abstract scored; strongest wins

The design is CHOSEN on the calibration claims (20% of SciFact train, never trained
on) and only then run on SciFact dev, so the test set does not pick the design.
Retrieval is fixed to hybrid, top 3 abstracts.

It also measures what the app's uncertainty labels actually deliver in this setting:
the temperature and 90% conformal threshold fitted in experiment 4 were fitted on oracle
evidence, so their coverage on retrieved evidence has to be measured, not assumed.

Writes results/e05b_evidence_design.json.
"""
import json

import numpy as np

from verity import conformal as C, data, metrics as M, pairs as P, retrieve as R
from verity.config import FINETUNED_NLI, L2I, LABELS, RESULTS
from verity.pairs import top_sentences
from verity.text import windows
from verity.verify import cached_logits, softmax

VERIFIERS = {"deberta-fever": "deberta-fever", "verity-finetuned": str(FINETUNED_NLI)}
DESIGNS = ["top3-concat", "per-sentence", "windows"]


def evidence_units(design, claim, sents):
    if design == "top3-concat":
        return [" ".join(sents[i] for i in top_sentences(claim, sents, 3))]
    if design == "per-sentence":
        return [sents[i] for i in top_sentences(claim, sents, 5)]
    return windows(sents, 2)


def calibration_of(vname):
    r = json.loads((RESULTS / "e04_conformal.json").read_text())["models"][vname]
    return r["temperature"], r["tests"]["scifact_dev"]["conformal"]["0.1"]["qhat"]


def run(claims, tag, corpus, retriever):
    ranked = retriever.search([c.text for c in claims], 3)
    y = [c.label for c in claims]
    out = {}
    for design in DESIGNS:
        pairs, owner = [], []
        for ci, (c, rk) in enumerate(zip(claims, ranked)):
            for did, _ in rk:
                for u in evidence_units(design, c.text, corpus[did].sentences):
                    pairs.append((u, c.text))
                    owner.append(ci)
        owner = np.array(owner)
        for vname, path in VERIFIERS.items():
            z = cached_logits(path, pairs, f"e5b_{tag}_{design}")
            p = softmax(z)
            t, q = calibration_of(vname)
            pt = softmax(z, t)
            pred, chosen = [], []
            for ci in range(len(claims)):
                idx = np.flatnonzero(owner == ci)
                b = idx[np.argmax(np.maximum(p[idx, 0], p[idx, 1]))]
                pred.append(LABELS[int(p[b].argmax())])
                chosen.append(pt[b])
            r = M.classification(y, pred, LABELS)
            r["macro_f1_ci95"] = M.bootstrap_ci(y, pred, M.macro_f1(LABELS))
            r["confusion"] = {t: {q: sum(1 for a, b in zip(y, pred) if a == t and b == q) for q in LABELS}
                              for t in LABELS}
            # the app's uncertainty labels, applied exactly as the pipeline applies them
            r["pipeline_conformal_at_90"] = {"temperature": t, "qhat": q,
                                             **C.summary(np.array(chosen), np.array([L2I[l] for l in y]), q)}
            out[f"{design} + {vname}"] = r
            print(f"{tag:12s} {design:13s} + {vname:17s} macro-F1 {r['macro_f1']:.3f}  "
                  f"recall S/R/N {r['per_class']['SUPPORTED']['recall']:.2f}/"
                  f"{r['per_class']['REFUTED']['recall']:.2f}/{r['per_class']['NEI']['recall']:.2f}", flush=True)
    return out


def main():
    corpus = data.scifact_corpus()
    retriever = R.build("hybrid", [(d.doc_id, f"{d.title}. {d.text}") for d in corpus.values()])
    _, cal_pairs = P.split_train_calibration(P.scifact_pairs("train"))
    cal_ids = {p["cid"] for p in cal_pairs}
    cal = [c for c in data.scifact_claims("train") if c.cid in cal_ids and c.label in LABELS]
    res = {"calibration": run(cal, "calibration", corpus, retriever)}
    chosen = max(res["calibration"], key=lambda k: res["calibration"][k]["macro_f1"])
    res["chosen_on_calibration"] = chosen
    print("chosen on calibration:", chosen, flush=True)
    dev = [c for c in data.scifact_claims("dev") if c.label in LABELS]
    res["dev"] = run(dev, "dev", corpus, retriever)
    (RESULTS / "e05b_evidence_design.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
