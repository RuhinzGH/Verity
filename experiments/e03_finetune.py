"""Experiment 3: fine-tune the verifier on SciFact (laptop GPU, 4 GB).

Starts from DeBERTa-v3-base already trained for NLI on MNLI/FEVER/ANLI and continues
training on SciFact *train* claims (80%, split by claim). The other 20% is never trained
on: it is the calibration set for experiment 4. SciFact dev is the test set, and it is not
looked at here -- the model kept is simply the one after the last epoch.

Writes models/verity-deberta-scifact/ and results/e03_finetune.json.
"""
import json
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from verity import pairs as P
from verity.config import FINETUNED_NLI, LABELS, NLI_MODELS, RESULTS, SEED

BASE = NLI_MODELS["deberta-fever"]
EPOCHS, LR, BATCH, ACCUM, MAXLEN = 3, 2e-5, 8, 2, 384


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    fit, cal = P.split_train_calibration(P.scifact_pairs("train", hard_negatives=True))
    print(f"fit pairs {len(fit)}  calibration pairs {len(cal)}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE)
    id2label = {int(i): l.lower() for i, l in model.config.id2label.items()}
    to_model = {"SUPPORTED": next(i for i, l in id2label.items() if l == "entailment"),
                "REFUTED": next(i for i, l in id2label.items() if l == "contradiction"),
                "NEI": next(i for i, l in id2label.items() if l in ("neutral", "not_entailment"))}
    model.cuda().train()
    model.gradient_checkpointing_enable()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(fit) + BATCH - 1) // BATCH) // ACCUM
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    log, t0 = [], time.time()
    for ep in range(EPOCHS):
        order = list(range(len(fit)))
        random.shuffle(order)
        tot, n = 0.0, 0
        for b, i in enumerate(range(0, len(order), BATCH)):
            batch = [fit[j] for j in order[i:i + BATCH]]
            enc = tok([p["evidence"] for p in batch], [p["claim"] for p in batch], truncation="only_first",
                      max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor([to_model[p["label"]] for p in batch], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc, labels=y).loss / ACCUM
            loss.backward()
            if (b + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item() * ACCUM * len(batch); n += len(batch)
        log.append({"epoch": ep + 1, "train_loss": tot / n, "minutes": round((time.time() - t0) / 60, 1)})
        print(log[-1], flush=True)

    model.eval()
    FINETUNED_NLI.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINETUNED_NLI)
    tok.save_pretrained(FINETUNED_NLI)
    info = {"base": BASE, "epochs": EPOCHS, "lr": LR, "batch": BATCH * ACCUM, "max_length": MAXLEN,
            "fit_pairs": len(fit), "calibration_pairs": len(cal),
            "fit_label_counts": {l: sum(p["label"] == l for p in fit) for l in LABELS},
            "gpu": torch.cuda.get_device_name(0),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "log": log}
    (RESULTS / "e03_finetune.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
