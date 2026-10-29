"""Experiment 2: which verifier labels a claim correctly when given the right evidence?

"Oracle evidence" isolates the verifier from retrieval: every claim comes with the
evidence an annotator marked (see verity/pairs.py).

Verifiers
  deberta-mnli      general NLI model (MNLI + SNLI), zero-shot
  deberta-fever     NLI model also trained on FEVER and ANLI, zero-shot
  verity-finetuned  deberta-fever further trained on SciFact train (experiment 3)
  qwen2.5-3b        LLM judge run locally on the laptop GPU (4-bit)
  qwen2.5-1.5b      smaller local LLM judge
  gpt-oss-120b      large LLM judge via the Groq API (only when GROQ_API_KEY is set)

Test sets
  SciFact dev       in-domain for the fine-tuned model
  Climate-FEVER     out of domain for every model; the LLM judges see a stratified
                    sample of 300 pairs (generation time), the NLI models see all 6,905

Writes results/e02_verifiers.json and per-pair predictions to results/e02_predictions.json.
"""
import json
import random
import sys

import numpy as np

from verity import metrics as M, pairs as P
from verity.config import FINETUNED_NLI, L2I, LABELS, LLM_JUDGE, LOCAL_LLM, LOCAL_LLM_LARGE, RESULTS, SEED
from verity.llm import api_available
from verity.verify import LLMJudge, cached_logits, softmax

NLI = {"deberta-mnli": "deberta-mnli", "deberta-fever": "deberta-fever"}
if FINETUNED_NLI.exists():
    NLI["verity-finetuned"] = str(FINETUNED_NLI)
LLMS = {"qwen2.5-3b": LOCAL_LLM_LARGE, "qwen2.5-1.5b": LOCAL_LLM}
if api_available():
    LLMS["gpt-oss-120b"] = LLM_JUDGE
SKIP_LLM = "--no-llm" in sys.argv


def stratified(pairs, n, seed=SEED):
    rng = random.Random(seed)
    by = {l: [p for p in pairs if p["label"] == l] for l in LABELS}
    out = []
    for l in LABELS:
        out += rng.sample(by[l], min(len(by[l]), n // 3))
    return out


def evaluate(name, pairs, probs):
    y = [p["label"] for p in pairs]
    pred = [LABELS[i] for i in probs.argmax(1)]
    r = M.classification(y, pred, LABELS)
    # resample whole claims: Climate-FEVER has five evidence pairs per claim
    r["macro_f1_ci95"] = M.bootstrap_ci(y, pred, M.macro_f1(LABELS), groups=[p["cid"] for p in pairs])
    r["ece"] = M.ece(probs, np.array([L2I[l] for l in y]))
    r["majority_baseline_accuracy"] = max(y.count(l) for l in LABELS) / len(y)
    print(f"  {name:18s} n={r['n']:5d}  acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f} "
          f"[{r['macro_f1_ci95'][0]:.3f}, {r['macro_f1_ci95'][1]:.3f}]  ECE {r['ece']:.3f}", flush=True)
    return r


def main():
    sets = {"scifact_dev": P.scifact_pairs("dev"), "climate_fever": P.climate_pairs()}
    sets["climate_fever_sample"] = stratified(sets["climate_fever"], 300)
    results, preds = {}, {}
    from verity.text import sentences
    for sname, pairs in sets.items():
        print(sname, len(pairs), {l: sum(p["label"] == l for p in pairs) for l in LABELS}, flush=True)
        results[sname], preds[sname] = {}, {"pairs": pairs}
        # format check: evidence length per label (a length gap would let a model shortcut)
        results[sname]["_evidence_format"] = {
            l: {"pairs": sum(p["label"] == l for p in pairs),
                "mean_sentences": float(np.mean([len(sentences(p["evidence"])) for p in pairs if p["label"] == l])),
                "mean_chars": float(np.mean([len(p["evidence"]) for p in pairs if p["label"] == l]))}
            for l in LABELS}
        ev = [(p["evidence"], p["claim"]) for p in pairs]
        for name, model in NLI.items():
            pr = softmax(cached_logits(model, ev, sname))
            results[sname][name] = evaluate(name, pairs, pr)
            preds[sname][name] = pr.tolist()
        if SKIP_LLM or sname == "climate_fever":
            continue
        for name, model in LLMS.items():
            if name == "qwen2.5-1.5b" and sname != "scifact_dev":    # weakest judge: in-domain only
                continue
            pr, js = LLMJudge(model).predict(ev)
            results[sname][name] = evaluate(name, pairs, pr)
            results[sname][name]["parse_errors"] = sum(bool(j.get("parse_error")) for j in js)
            preds[sname][name] = pr.tolist()
            preds[sname][name + "_raw"] = js
    (RESULTS / "e02_verifiers.json").write_text(json.dumps(results, indent=1))
    (RESULTS / "e02_predictions.json").write_text(json.dumps(preds))


if __name__ == "__main__":
    main()
