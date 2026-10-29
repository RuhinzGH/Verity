"""Experiment 3b: how much do the fine-tuning results depend on luck, and does
re-weighting the classes fix the lean towards "not enough evidence"?

Trains the verifier recipe of experiment 3 with three seeds, plus a class-balanced
variant (loss weighted by inverse class frequency) with the same three seeds. Each model
is scored on the held-out calibration claims and on SciFact dev (oracle evidence).

The deployed model is NOT replaced: models are trained into a scratch folder and
deleted after scoring. The comparison between the two recipes is made on the calibration
claims; dev numbers are reported for transparency, not used to choose.

Writes results/e03b_seeds.json.
"""
import json
import random
import shutil
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from verity import metrics as M, pairs as P
from verity.config import L2I, LABELS, MODELS, NLI_MODELS, RESULTS
from verity.verify import NLIVerifier

BASE = NLI_MODELS["deberta-fever"]
EPOCHS, LR, BATCH, ACCUM, MAXLEN = 3, 2e-5, 8, 2, 384
SEEDS = [42, 7, 1234]
SCRATCH = MODELS / "scratch_seed_run"


def train(fit, seed, balanced):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE)
    id2label = {int(i): l.lower() for i, l in model.config.id2label.items()}
    to_model = {"SUPPORTED": next(i for i, l in id2label.items() if l == "entailment"),
                "REFUTED": next(i for i, l in id2label.items() if l == "contradiction"),
                "NEI": next(i for i, l in id2label.items() if l in ("neutral", "not_entailment"))}
    weight = None
    if balanced:
        counts = {l: sum(p["label"] == l for p in fit) for l in LABELS}
        w = torch.zeros(len(id2label))
        for l, c in counts.items():
            w[to_model[l]] = len(fit) / (len(LABELS) * c)
        weight = w.cuda()
    model.cuda().train()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(fit) + BATCH - 1) // BATCH) // ACCUM
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for _ in range(EPOCHS):
        order = list(range(len(fit)))
        random.shuffle(order)
        for b, i in enumerate(range(0, len(order), BATCH)):
            batch = [fit[j] for j in order[i:i + BATCH]]
            enc = tok([p["evidence"] for p in batch], [p["claim"] for p in batch], truncation="only_first",
                      max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor([to_model[p["label"]] for p in batch], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, y, weight=weight) / ACCUM
            loss.backward()
            if (b + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
    model.eval()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    model.save_pretrained(SCRATCH)
    tok.save_pretrained(SCRATCH)
    del model
    torch.cuda.empty_cache()


def score(pairs):
    v = NLIVerifier(str(SCRATCH))
    probs = v.predict([(p["evidence"], p["claim"]) for p in pairs])
    del v
    torch.cuda.empty_cache()
    y = [p["label"] for p in pairs]
    pred = [LABELS[i] for i in probs.argmax(1)]
    r = M.classification(y, pred, LABELS)
    r["ece"] = M.ece(probs, np.array([L2I[l] for l in y]))
    return r


def main():
    fit, _ = P.split_train_calibration(P.scifact_pairs("train", hard_negatives=True))
    _, cal = P.split_train_calibration(P.scifact_pairs("train"))
    dev = P.scifact_pairs("dev")
    # runs are saved one by one, so an interrupted experiment resumes where it stopped
    partial = RESULTS / "e03b_runs_partial.json"
    runs = json.loads(partial.read_text()) if partial.exists() else []
    for balanced in (False, True):
        for seed in SEEDS:
            recipe = "class-balanced" if balanced else "standard"
            if any(r["recipe"] == recipe and r["seed"] == seed for r in runs):
                continue
            t0 = time.time()
            train(fit, seed, balanced)
            rc, rd = score(cal), score(dev)
            runs.append({"recipe": "class-balanced" if balanced else "standard", "seed": seed,
                         "calibration_macro_f1": rc["macro_f1"], "dev_macro_f1": rd["macro_f1"],
                         "dev_accuracy": rd["accuracy"], "dev_ece": rd["ece"],
                         "dev_recall": {l: rd["per_class"][l]["recall"] for l in LABELS},
                         "minutes": round((time.time() - t0) / 60, 1)})
            print(runs[-1], flush=True)
            partial.write_text(json.dumps(runs, indent=1))
    shutil.rmtree(SCRATCH, ignore_errors=True)
    runs.sort(key=lambda r: (r["recipe"] != "standard", SEEDS.index(r["seed"])))
    summary = {}
    for recipe in ("standard", "class-balanced"):
        rs = [r for r in runs if r["recipe"] == recipe]
        summary[recipe] = {k: {"mean": float(np.mean([r[k] for r in rs])), "sd": float(np.std([r[k] for r in rs], ddof=1))}
                           for k in ("calibration_macro_f1", "dev_macro_f1", "dev_accuracy", "dev_ece")}
        summary[recipe]["dev_recall_mean"] = {l: float(np.mean([r["dev_recall"][l] for r in rs])) for l in LABELS}
    better = max(summary, key=lambda k: summary[k]["calibration_macro_f1"]["mean"])
    out = {"seeds": SEEDS, "runs": runs, "summary": summary, "better_on_calibration": better,
           "note": "The deployed model (experiment 3, seed 42, standard recipe) was not replaced."}
    (RESULTS / "e03b_seeds.json").write_text(json.dumps(out, indent=1))
    print("better on calibration:", better, summary)


if __name__ == "__main__":
    main()
