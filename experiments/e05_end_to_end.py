"""Experiment 5: the full pipeline on SciFact dev -- retrieve, select, verify.

No gold evidence this time. For each claim:
  retrieve the top 3 abstracts (BM25, bge-base or hybrid) from all 5,183,
  keep each abstract's 3 sentences closest to the claim,
  verify every (sentences, claim) pair,
  take the label of the abstract with the strongest SUPPORTED or REFUTED probability.

Compared with the oracle-evidence score from experiment 2, the gap is the cost of the
system finding and cutting its own evidence. Retrieval itself usually succeeds (the
evidence abstract is in the top 3 for most claims), so experiment 5b looks at how the
evidence is cut. Writes results/e05_end_to_end.json.
"""
import json

import numpy as np

from verity import data, metrics as M, retrieve as R
from verity.config import FINETUNED_NLI, LABELS, RESULTS
from verity.pairs import top_sentences
from verity.verify import cached_logits, softmax

RETRIEVERS = ["bm25", "bge-base", "hybrid"]
VERIFIERS = {"deberta-fever": "deberta-fever", "verity-finetuned": str(FINETUNED_NLI)}
TOP_DOCS = 3


def main():
    corpus = data.scifact_corpus()
    claims = [c for c in data.scifact_claims("dev") if c.label in LABELS]
    passages = [(d.doc_id, f"{d.title}. {d.text}") for d in corpus.values()]
    y = [c.label for c in claims]
    out = {"n_claims": len(claims), "top_docs": TOP_DOCS, "runs": {}}
    for rname in RETRIEVERS:
        ranked = R.build(rname, passages).search([c.text for c in claims], TOP_DOCS)
        pairs, owner = [], []
        for ci, (c, rk) in enumerate(zip(claims, ranked)):
            for did, _ in rk:
                sents = corpus[did].sentences
                pick = top_sentences(c.text, sents, 3)
                pairs.append((" ".join(sents[i] for i in pick), c.text))
                owner.append(ci)
        owner = np.array(owner)
        hit = np.mean([bool(set(c.evidence) & {d for d, _ in rk}) for c, rk in zip(claims, ranked) if c.evidence])
        for vname, path in VERIFIERS.items():
            p = softmax(cached_logits(path, pairs, f"e2e_{rname}"))
            pred = []
            for ci in range(len(claims)):
                pc = p[owner == ci]
                best = pc[np.argmax(np.maximum(pc[:, 0], pc[:, 1]))]
                pred.append(LABELS[int(best.argmax())])
            r = M.classification(y, pred, LABELS)
            r["macro_f1_ci95"] = M.bootstrap_ci(y, pred, M.macro_f1(LABELS))
            r["evidence_hit@3"] = float(hit)
            out["runs"][f"{rname} + {vname}"] = r
            print(f"{rname:8s} + {vname:17s} acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f} "
                  f"[{r['macro_f1_ci95'][0]:.3f}, {r['macro_f1_ci95'][1]:.3f}]  evidence hit@3 {hit:.3f}",
                  flush=True)
    (RESULTS / "e05_end_to_end.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
