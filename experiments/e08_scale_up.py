"""Experiment 8: what changes with a big cloud GPU instead of a 4 GB laptop?

Run on a large GPU (Colab A100/L4; see colab/verity_cloud_run.ipynb). Same data, splits,
length-matched pairs, prompts and metrics as experiments 2-3, only bigger models:

  A. Verifier size: DeBERTa-v3-LARGE (435M) fine-tuned on the same SciFact pairs as
     experiment 3 (3 epochs, but learning rate 1e-5 and batch 16, the usual settings for
     the large model), 3 seeds; compared with the laptop's DeBERTa-v3-base.
  B. LLM judge size: Qwen2.5-7B, -14B and -32B-Instruct (bf16, no quantisation) with the
     same judge prompt as experiment 2, on SciFact dev and the Climate-FEVER sample.
  C. A 32B answer-judge on RAGTruth: the same whole-answer prompt and the same 240 test
     answers as experiment 6's local judge, to compare against claim-by-claim checking.

    --only=large | judges (7B, 14B) | judge32 | ragtruth32 | check14 | rescore

check14 re-reads the 14B judge's cached answers and regenerates 50 unbatched (it exposed
a scoring bug in verify.judgement_probs); rescore re-scores every judge from its cached
answers after that fix, without generating anything.

Writes results/e08_scale_up.json and prints it between RESULTS-JSON markers, so the
numbers can be read back from the notebook's output.
"""
import json
import random
import shutil
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from verity import metrics as M, pairs as P
from verity.config import L2I, LABELS, MODELS, RESULTS, SEED
from verity.verify import LLMJudge, NLIVerifier

LARGE = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
EPOCHS, LR, BATCH, MAXLEN = 3, 1e-5, 16, 384
SEEDS = [42, 7, 1234]
JUDGES = {"qwen2.5-7b": "local:Qwen/Qwen2.5-7B-Instruct", "qwen2.5-14b": "local:Qwen/Qwen2.5-14B-Instruct"}
JUDGE_32B = {"qwen2.5-32b": "local:Qwen/Qwen2.5-32B-Instruct"}
SCRATCH = MODELS / "scratch_large"


def stratified(pairs, n, seed=SEED):          # same sample as experiment 2
    rng = random.Random(seed)
    out = []
    for l in LABELS:
        by = [p for p in pairs if p["label"] == l]
        out += rng.sample(by, min(len(by), n // 3))
    return out


def evaluate(pairs, probs):
    y = [p["label"] for p in pairs]
    pred = [LABELS[i] for i in probs.argmax(1)]
    r = M.classification(y, pred, LABELS)
    r["macro_f1_ci95"] = M.bootstrap_ci(y, pred, M.macro_f1(LABELS), groups=[p["cid"] for p in pairs])
    r["ece"] = M.ece(probs, np.array([L2I[l] for l in y]))
    return r


def train_large(fit, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(LARGE)
    model = AutoModelForSequenceClassification.from_pretrained(LARGE)
    id2label = {int(i): l.lower() for i, l in model.config.id2label.items()}
    to_model = {"SUPPORTED": next(i for i, l in id2label.items() if l == "entailment"),
                "REFUTED": next(i for i, l in id2label.items() if l == "contradiction"),
                "NEI": next(i for i, l in id2label.items() if l in ("neutral", "not_entailment"))}
    model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(fit) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for _ in range(EPOCHS):
        order = list(range(len(fit)))
        random.shuffle(order)
        for i in range(0, len(order), BATCH):
            batch = [fit[j] for j in order[i:i + BATCH]]
            enc = tok([p["evidence"] for p in batch], [p["claim"] for p in batch], truncation="only_first",
                      max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor([to_model[p["label"]] for p in batch], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    model.eval()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    model.save_pretrained(SCRATCH)
    tok.save_pretrained(SCRATCH)
    del model
    torch.cuda.empty_cache()


def main():
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    gpu = torch.cuda.get_device_name(0)
    fit, _ = P.split_train_calibration(P.scifact_pairs("train", hard_negatives=True))
    _, cal = P.split_train_calibration(P.scifact_pairs("train"))
    dev, climate = P.scifact_pairs("dev"), P.climate_pairs()
    sample = stratified(climate, 300)
    path = RESULTS / "e08_scale_up.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out.update({"gpu": gpu, "large_verifier": LARGE})

    if not only or "large" in only:
        runs = []
        for seed in SEEDS:
            t0 = time.time()
            train_large(fit, seed)
            v = NLIVerifier(str(SCRATCH), batch_size=32)
            res = {"seed": seed}
            for name, pairs in [("calibration", cal), ("scifact_dev", dev), ("climate_fever", climate),
                                ("climate_fever_sample", sample)]:
                res[name] = evaluate(pairs, v.predict([(p["evidence"], p["claim"]) for p in pairs]))
            res["minutes"] = round((time.time() - t0) / 60, 1)
            del v
            torch.cuda.empty_cache()
            runs.append(res)
            print({"seed": seed, **{k: round(res[k]["macro_f1"], 3) for k in ("calibration", "scifact_dev", "climate_fever")}},
                  flush=True)
        shutil.rmtree(SCRATCH, ignore_errors=True)
        out["large_runs"] = runs
        out["large_summary"] = {k: {"mean": float(np.mean([r[k]["macro_f1"] for r in runs])),
                                    "sd": float(np.std([r[k]["macro_f1"] for r in runs], ddof=1))}
                                for k in ("calibration", "scifact_dev", "climate_fever", "climate_fever_sample")}

    todo = {}
    if not only or "judges" in only:
        todo.update(JUDGES)
    if not only or "judge32" in only:
        todo.update(JUDGE_32B)
    if todo:
        out.setdefault("judges", {})
        for name, model in todo.items():
            t0 = time.time()
            out["judges"][name] = {}
            for sname, pairs in [("scifact_dev", dev), ("climate_fever_sample", sample)]:
                probs, js = LLMJudge(model).predict([(p["evidence"], p["claim"]) for p in pairs])
                r = evaluate(pairs, probs)
                r["parse_errors"] = sum(bool(j.get("parse_error")) for j in js)
                r["predicted_counts"] = {l: sum(j["label"] == l for j in js) for l in LABELS}
                r["labels"] = [j["label"] for j in js]          # kept so the run can be audited
                out["judges"][name][sname] = r
                print(name, sname, round(r["macro_f1"], 3), r["predicted_counts"], flush=True)
            out["judges"][name]["minutes"] = round((time.time() - t0) / 60, 1)
            from verity import llm
            llm._local_model.cache_clear()
            torch.cuda.empty_cache()

    if "rescore" in only:
        # Re-score every judge from its cached raw answers with the corrected
        # judgement_probs (no generation): the earlier scoring let low stated confidences
        # tie all three labels, and the tie was scored as SUPPORTED.
        from verity import llm
        from verity.verify import JUDGE_PROMPT, judgement_probs, parse_judgement
        for name, model in {**JUDGES, **JUDGE_32B}.items():
            res = {}
            for sname, pairs in [("scifact_dev", dev), ("climate_fever_sample", sample)]:
                raws = [llm._cache().get(llm._key(model, JUDGE_PROMPT.format(evidence=p["evidence"][:6000],
                                                                             claim=p["claim"]), 600)) for p in pairs]
                if any(r is None for r in raws):
                    print(name, sname, "missing cached answers:", sum(r is None for r in raws), flush=True)
                    continue
                js = [parse_judgement(r) for r in raws]
                r = evaluate(pairs, np.array([judgement_probs(j) for j in js]))
                r["parse_errors"] = sum(bool(j.get("parse_error")) for j in js)
                r["predicted_counts"] = {l: sum(j["label"] == l for j in js) for l in LABELS}
                r["labels"] = [j["label"] for j in js]
                res[sname] = r
                print(name, sname, "rescored macro-F1", round(r["macro_f1"], 3), flush=True)
            if res:
                out.setdefault("judges", {}).setdefault(name, {}).update(res)
                out["judges"][name]["rescored"] = True

    if "check14" in only:
        # The 14B judge scored far below the 7B with an almost identical label mix, which
        # looks like a bug, not a result. Re-read its raw cached answers, and regenerate 50
        # claims one at a time (no batching) to see whether batched generation is at fault.
        from verity import llm
        from verity.verify import JUDGE_PROMPT, parse_judgement
        chk = {}
        for name in ("qwen2.5-7b", "qwen2.5-14b"):
            model = JUDGES[name]
            prompts = [JUDGE_PROMPT.format(evidence=p["evidence"][:6000], claim=p["claim"]) for p in dev]
            raws = [llm._cache().get(llm._key(model, q, 600)) for q in prompts]
            got = [(p, parse_judgement(r)) for p, r in zip(dev, raws) if r is not None]
            acc = float(np.mean([j["label"] == p["label"] for p, j in got])) if got else None
            chk[name] = {"cached_answers": len(got), "accuracy_from_cache": acc,
                         "examples": [{"gold": p["label"], "said": j["label"], "claim": p["claim"][:120],
                                       "raw": r[:300]} for (p, j), r in list(zip(got, [r for r in raws if r]))[:6]]}
            print(name, "cached", len(got), "accuracy", acc, flush=True)
        model = JUDGES["qwen2.5-14b"]
        L = llm.LLM(model)
        single, same = [], 0
        for p in dev[:50]:
            q = JUDGE_PROMPT.format(evidence=p["evidence"][:6000], claim=p["claim"])
            j1 = parse_judgement(L._local(q, 600))                       # unbatched
            jb = parse_judgement(llm._cache().get(llm._key(model, q, 600)) or "")
            single.append(j1["label"] == p["label"])
            same += j1["label"] == jb["label"]
        chk["qwen2.5-14b_unbatched_50"] = {"accuracy": float(np.mean(single)), "agrees_with_batched": same / 50}
        print("14b unbatched on 50:", chk["qwen2.5-14b_unbatched_50"], flush=True)
        out["check_14b"] = chk
        llm._local_model.cache_clear()
        torch.cuda.empty_cache()

    if not only or "ragtruth32" in only:
        from experiments.e06_ragtruth import llm_judge, prf, sample
        test = sample("test")
        y = [r.hallucinated for r in test]
        t0 = time.time()
        flags = llm_judge(test, JUDGE_32B["qwen2.5-32b"])
        resp = [any(f) for f in flags]
        yt, fl = np.asarray(y), np.asarray(resp)
        rng = np.random.default_rng(SEED)
        boot = [prf(yt[i], fl[i])["f1"] for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
        out["ragtruth_judge_32b"] = {
            "response": {**prf(y, resp), "f1_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]},
            "clean_answers_flagged": float(fl[~yt].mean()),
            "by_task": {t: prf([a for a, r in zip(y, test) if r.task == t], [b for b, r in zip(resp, test) if r.task == t])
                        for t in ["QA", "Summary", "Data2txt"]},
            "n_test": len(test), "minutes": round((time.time() - t0) / 60, 1)}
        print("ragtruth 32b judge", out["ragtruth_judge_32b"]["response"], flush=True)

    path.write_text(json.dumps(out, indent=1))
    print("RESULTS-JSON-START")
    print(json.dumps(out))
    print("RESULTS-JSON-END")


if __name__ == "__main__":
    main()
