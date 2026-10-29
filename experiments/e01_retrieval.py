"""Experiment 1: which retriever finds the evidence?

SciFact dev   188 claims with gold evidence, searched over 5,183 abstracts
Climate-FEVER 907 claims with a supporting or refuting sentence, searched over the
              pooled 5,240 evidence sentences

Writes results/e01_retrieval.json.
"""
import json
import time

import numpy as np

from verity import data, metrics as M, retrieve as R
from verity.config import RESULTS

METHODS = ["bm25", "bge-small", "bge-base", "hybrid", "hybrid+rerank"]
KS = [1, 3, 5, 10]


def run(name, claims, passages):
    claims = [c for c in claims if c.evidence]
    queries = [c.text for c in claims]
    gold = [set(c.evidence) for c in claims]
    out = {}
    for m in METHODS:
        t0 = time.perf_counter()
        r = R.build(m, passages)
        t1 = time.perf_counter()
        runs = r.search(queries, k=10)
        t2 = time.perf_counter()
        ranked = [[pid for pid, _ in x] for x in runs]
        res = {f"recall@{k}": float(np.mean([M.recall_at(rk, g, k) for rk, g in zip(ranked, gold)])) for k in KS}
        res.update({f"hit@{k}": float(np.mean([M.hit_at(rk, g, k) for rk, g in zip(ranked, gold)])) for k in KS})
        res["mrr@10"] = float(np.mean([M.mrr(rk, g) for rk, g in zip(ranked, gold)]))
        res["ndcg@10"] = float(np.mean([M.ndcg(rk, g) for rk, g in zip(ranked, gold)]))
        res["index_seconds"] = round(t1 - t0, 2)
        res["ms_per_query"] = round(1000 * (t2 - t1) / len(queries), 2)
        out[m] = res
        print(f"{name:14s} {m:14s} R@1 {res['recall@1']:.3f}  R@5 {res['recall@5']:.3f}  "
              f"R@10 {res['recall@10']:.3f}  MRR {res['mrr@10']:.3f}  nDCG {res['ndcg@10']:.3f}  "
              f"{res['ms_per_query']} ms/q", flush=True)
    return {"n_claims": len(claims), "n_passages": len(passages), "methods": out}


if __name__ == "__main__":
    corpus = data.scifact_corpus()
    sf = [(d.doc_id, f"{d.title}. {d.text}") for d in corpus.values()]
    cf_claims, cf_corpus = data.climate_fever()
    cf = [(d.doc_id, f"{d.title}: {d.text}") for d in cf_corpus.values()]
    result = {"scifact_dev": run("SciFact", data.scifact_claims("dev"), sf),
              "climate_fever": run("Climate-FEVER", cf_claims, cf)}
    (RESULTS / "e01_retrieval.json").write_text(json.dumps(result, indent=1))
