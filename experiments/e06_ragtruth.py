"""Experiment 6: catching hallucinations in real LLM answers (RAGTruth).

RAGTruth gives an LLM a source context (news to summarise, web passages for a question,
or structured business data to describe) and a human marks every span of the answer
that conflicts with, or is not backed by, the context.

Sample: 40 hallucinated + 40 clean answers per task type (240 test answers), and the
same from the training split for choosing the decision threshold.

Verity per answer:
  decompose the answer (sentences | local Qwen2.5-1.5B | gpt-oss-20b when a key is set)
  for each claim, retrieve the 3 closest 2-sentence windows of the context (hybrid)
  support(claim) = highest P(SUPPORTED) over those windows
  hallucination score of the answer = 1 - lowest support over its claims
Flag the answer if the score exceeds a threshold chosen on the training sample (best F1).
A sentence is flagged if any of its claims is; it is truly hallucinated if it overlaps
an annotated span.

Baseline: an LLM judge (local Qwen2.5-3B; gpt-oss-120b when a key is set) reads the context and the numbered answer and lists the sentences
that are not supported (one call per answer, no decomposition, no threshold).

Writes results/e06_ragtruth.json.
"""
import json
import random
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from verity import decompose as D, retrieve as R
from verity.config import FINETUNED_NLI, LLM_DECOMPOSER, LLM_JUDGE, LOCAL_LLM, LOCAL_LLM_LARGE, SEED
from verity.config import RESULTS
from verity.data import ragtruth
from verity.llm import LLM, api_available
from verity.text import sentence_spans, sentences, windows
from verity.verify import cached_logits, softmax

PER_CELL = 40
DECOMPOSERS = {"sentences": None, "qwen2.5-1.5b": LOCAL_LLM}
JUDGES = {"qwen2.5-3b": LOCAL_LLM_LARGE}
if api_available():
    DECOMPOSERS["gpt-oss-20b"] = LLM_DECOMPOSER
    JUDGES["gpt-oss-120b"] = LLM_JUDGE
VERIFIERS = {"deberta-fever": "deberta-fever", "verity-finetuned": str(FINETUNED_NLI)}
ONLY = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]


def sample(split):
    rng = random.Random(SEED)
    rs = ragtruth(split)
    out = []
    for task in ["QA", "Summary", "Data2txt"]:
        for h in (True, False):
            cell = [r for r in rs if r.task == task and r.hallucinated == h]
            out += rng.sample(cell, PER_CELL)
    return out


def sentence_truth(r):
    """Per sentence of the answer: does it overlap a hallucinated span?"""
    spans = sentence_spans(r.output)
    return [any(a < s["end"] and s["start"] < e for s in r.spans) for _, a, e in spans]


def claims_for(r, dname):
    if DECOMPOSERS[dname] is None:
        return D.by_sentence(r.output)
    return D.by_llm(r.output, DECOMPOSERS[dname])


def score(responses, dname, vpath, tag):
    """Per answer: (answer score, per-sentence scores) where score = 1 - support."""
    out = []
    for r in responses:
        cs = claims_for(r, dname)
        passages = [(str(i), w) for i, w in enumerate(windows(sentences(r.context), 2))] or [("0", r.context)]
        n_s = len(sentence_spans(r.output))
        if not cs:
            out.append((0.0, [0.0] * n_s))
            continue
        if len(passages) > 3:
            ranked = R.Hybrid(passages, depth=len(passages)).search([c.text for c in cs], 3)
        else:
            ranked = [[(pid, 0.0) for pid, _ in passages] for _ in cs]
        text = dict(passages)
        pairs = [(text[pid], c.text) for c, rk in zip(cs, ranked) for pid, _ in rk]
        p = softmax(cached_logits(vpath, pairs, f"rt_{tag}_{r.rid}_{dname}"))
        sup, i = [], 0
        for rk in ranked:
            sup.append(float(p[i:i + len(rk), 0].max()))
            i += len(rk)
        sent = [0.0] * n_s
        for c, s in zip(cs, sup):
            if c.sentence < n_s:
                sent[c.sentence] = max(sent[c.sentence], 1 - s)
        out.append((1 - min(sup), sent))
    return out


def prf(t, p):
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    tp = (t & p).sum()
    prec = tp / p.sum() if p.sum() else 0.0
    rec = tp / t.sum() if t.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def best_threshold(scores, truth):
    cands = sorted(set(scores))
    return max(cands, key=lambda th: prf(truth, [s >= th for s in scores])["f1"])


JUDGE = """You are checking an AI-written answer against its SOURCE. A sentence is unsupported
if any part of it contradicts the source or adds information the source does not contain.

SOURCE:
{context}

ANSWER (sentences numbered):
{numbered}

Reply with JSON only: {{"unsupported": [<sentence numbers>]}}  (an empty list if all are supported)"""


def llm_judge(responses, model=LLM_JUDGE):
    prompts, n_spans = [], []
    for r in responses:
        spans = sentence_spans(r.output)
        numbered = "\n".join(f"[{i + 1}] {s}" for i, (s, _, _) in enumerate(spans))
        ctx = r.context[:12000] if not model.startswith("local:") else r.context[:6000]
        prompts.append(JUDGE.format(context=ctx, numbered=numbered))
        n_spans.append(len(spans))
    raws = LLM(model).complete_many(prompts, max_tokens=300, batch_size=4, progress=f"answer judge {model}")
    out = []
    for raw, n in zip(raws, n_spans):
        try:
            bad = {int(x) - 1 for x in json.loads(raw[raw.find("{"):raw.rfind("}") + 1])["unsupported"]}
        except (ValueError, KeyError, TypeError):
            bad = set()
        out.append([i in bad for i in range(n)])
    return out


def main():
    train, test = sample("train"), sample("test")
    y_tr = [r.hallucinated for r in train]
    y_te = [r.hallucinated for r in test]
    tasks = [r.task for r in test]
    sent_truth = [sentence_truth(r) for r in test]
    res = {"n_test": len(test), "n_train": len(train), "methods": {},
           "test_ids": [r.rid for r in test], "test_truth": y_te, "test_tasks": tasks, "per_answer": {}}
    rng = np.random.default_rng(SEED)
    boot = [rng.integers(0, len(y_te), len(y_te)) for _ in range(1000)]
    yt = np.asarray(y_te)

    def ci(fn):
        v = [fn(i) for i in boot]
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    def report(name, flags, scores=None, sent_flags=None, th=None):
        m = {"response": prf(y_te, flags), "threshold": th,
             "by_task": {t: prf([y for y, k in zip(y_te, tasks) if k == t],
                                [f for f, k in zip(flags, tasks) if k == t]) for t in ["QA", "Summary", "Data2txt"]}}
        fl = np.asarray(flags, bool)
        m["response"]["f1_ci95"] = ci(lambda i: prf(yt[i], fl[i])["f1"])
        m["share_flagged"] = float(fl.mean())
        m["clean_answers_flagged"] = float(fl[~yt].mean())
        res["per_answer"][name] = {"flag": fl.tolist()}
        if scores is not None:
            sc = np.asarray(scores, float)
            m["auroc"] = float(roc_auc_score(y_te, scores))
            m["auroc_ci95"] = ci(lambda i: roc_auc_score(yt[i], sc[i]) if 0 < yt[i].sum() < len(i) else np.nan)
            res["per_answer"][name]["score"] = sc.tolist()
        if sent_flags is not None:
            st = [x for s in sent_truth for x in s]
            sf = [x for s in sent_flags for x in s]
            m["sentence"] = prf(st, sf)
        res["methods"][name] = m
        print(f"{name:34s} F1 {m['response']['f1']:.3f}  P {m['response']['precision']:.3f}  "
              f"R {m['response']['recall']:.3f}  AUROC {m.get('auroc', float('nan')):.3f}  "
              f"sentence-F1 {m.get('sentence', {}).get('f1', float('nan')):.3f}  "
              + "  ".join(f"{t} {v['f1']:.2f}" for t, v in m["by_task"].items()), flush=True)

    for dname in DECOMPOSERS:
        if ONLY and dname not in ONLY:
            continue
        if DECOMPOSERS[dname] is not None:
            D.prefetch([r.output for r in train + test], DECOMPOSERS[dname])
        for vname, vpath in VERIFIERS.items():
            tr = score(train, dname, vpath, "train")
            te = score(test, dname, vpath, "test")
            th = best_threshold([s for s, _ in tr], y_tr)
            report(f"{dname} + {vname}", [s >= th for s, _ in te], [s for s, _ in te],
                   [[x >= th for x in sent] for _, sent in te], th)
    if not ONLY or "judge" in ONLY:
        for jname, jmodel in JUDGES.items():
            sj = llm_judge(test, jmodel)
            report(f"{jname} judge (no decomposition)", [any(s) for s in sj], None, sj)
    res["always_flag_baseline"] = prf(y_te, [True] * len(y_te))
    res["always_flag_baseline"]["f1_ci95"] = ci(lambda i: prf(yt[i], np.ones(len(i), bool))["f1"])
    # partial runs (--only=...) merge their methods into the shared results file
    path = RESULTS / "e06_ragtruth.json"
    if ONLY and path.exists():
        prev = json.loads(path.read_text())
        prev["methods"].update(res["methods"])
        res["methods"] = prev["methods"]
        prev.setdefault("per_answer", {}).update(res["per_answer"])
        res["per_answer"] = prev["per_answer"]
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
