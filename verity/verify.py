"""Claim verification: does this evidence support the claim, refute it, or neither?

Two families:
  NLIVerifier  a DeBERTa cross-encoder trained for natural language inference; reads
               (evidence, claim) and returns probabilities for entailment / contradiction /
               neutral, mapped to SUPPORTED / REFUTED / NEI
  LLMJudge     a large language model asked the same question in a prompt, answering
               in JSON with a label, a quoted evidence span and a stated confidence
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import CACHE, LABELS, LLM_JUDGE, NLI_MODELS

_MAP = {"entailment": "SUPPORTED", "contradiction": "REFUTED", "neutral": "NEI",
        "not_entailment": "NEI", "supported": "SUPPORTED", "refuted": "REFUTED", "nei": "NEI"}


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def cpu_limit() -> int:
    """CPUs this process may really use. A container (such as the live app's) sees every
    core of its host but gets only a share of them; torch would start one thread per
    visible core and the threads would then wait on each other."""
    import os
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for files in (("/sys/fs/cgroup/cpu.max",),                     # cgroup v2: "200000 100000"
                  ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):  # v1
        try:
            vals = " ".join(open(f).read() for f in files).split()
            quota, period = int(vals[0]), int(vals[1])
        except (OSError, ValueError, IndexError):         # missing file, or "max" (no quota)
            continue
        if quota > 0:
            n = min(n, max(1, -(-quota // period)))
    return n


class NLIVerifier:
    def __init__(self, name_or_path: str, max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if torch.get_num_threads() > cpu_limit():
            torch.set_num_threads(cpu_limit())
        path = NLI_MODELS.get(name_or_path, name_or_path)
        self.name = Path(str(name_or_path)).name
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(_device()).eval()
        if _device() == "cuda":
            self.model.half()
        id2label = {int(i): l.lower() for i, l in self.model.config.id2label.items()}
        # column order that turns the model's output into LABELS order
        self.cols = [next(i for i, l in id2label.items() if _MAP.get(l) == want) for want in LABELS]
        self.max_length, self.batch_size, self.torch = max_length, batch_size, torch

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """pairs: (evidence, claim). Returns an (n, 3) array of probabilities in LABELS order."""
        return softmax(self.logits(pairs))

    def logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Raw logits in LABELS order (used for temperature scaling).

        Pairs are batched shortest first, so a batch is padded only to similar lengths
        (about a third faster on a CPU); the output is returned in the input order."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        with self.torch.inference_mode():
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                ev, cl = zip(*(pairs[j] for j in idx))
                enc = self.tok(list(ev), list(cl), truncation="only_first", max_length=self.max_length,
                               padding=True, return_tensors="pt").to(_device())
                out[idx] = self.model(**enc).logits.float()[:, self.cols].cpu().numpy()
        return out


@lru_cache(maxsize=None)
def nli(name: str) -> NLIVerifier:
    return NLIVerifier(name)


# ------------------------------------------------------------------ cached NLI scores
def cached_logits(model: str, pairs: list[tuple[str, str]], tag: str) -> np.ndarray:
    h = hashlib.sha1(json.dumps([model, pairs]).encode("utf-8")).hexdigest()[:16]
    path = CACHE / f"nli_{tag}_{Path(model).name}_{h}.npy"
    if path.exists():
        return np.load(path)
    v = nli(model).logits(pairs)
    np.save(path, v)
    return v


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ LLM judge
JUDGE_PROMPT = """You are a careful fact-checker. Decide whether the EVIDENCE supports the CLAIM.

Answer SUPPORTED only if the evidence states or directly implies the claim.
Answer REFUTED only if the evidence states or directly implies that the claim is false.
Otherwise answer NEI (not enough information) - including when the evidence is on the
topic but does not settle the claim. Use only the evidence, not outside knowledge.

EVIDENCE:
{evidence}

CLAIM:
{claim}

Reply with JSON only:
{{"label": "SUPPORTED" | "REFUTED" | "NEI", "quote": "<shortest evidence span that decides it, or empty>", "confidence": <number from 0 to 1>}}"""


class LLMJudge:
    def __init__(self, model: str = LLM_JUDGE):
        from .llm import LLM
        self.name = model.split("/")[-1]
        self.llm = LLM(model)

    def judge(self, evidence: str, claim: str) -> dict:
        raw = self.llm.complete(JUDGE_PROMPT.format(evidence=evidence[:6000], claim=claim), max_tokens=600)
        return parse_judgement(raw)

    def predict(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, list[dict]]:
        """Pseudo-probabilities from the stated confidence: the chosen label gets c,
        the other two share 1 - c. Returns (probs, raw judgements)."""
        prompts = [JUDGE_PROMPT.format(evidence=e[:6000], claim=c) for e, c in pairs]
        raws = self.llm.complete_many(prompts, max_tokens=600, progress=f"judge {self.name}")
        js = [parse_judgement(r) for r in raws]
        return np.array([judgement_probs(j) for j in js]), js


def parse_judgement(raw: str) -> dict:
    txt = raw.strip()
    s, e = txt.find("{"), txt.rfind("}")
    try:
        j = json.loads(txt[s:e + 1])
    except (ValueError, json.JSONDecodeError):
        return {"label": _label_from_prose(txt), "quote": "", "confidence": 0.5, "parse_error": True}
    lab = str(j.get("label", "NEI")).upper().strip()
    lab = _MAP.get(lab.lower(), lab if lab in LABELS else "NEI")
    try:
        conf = float(j.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    if not math.isfinite(conf):              # json accepts NaN / Infinity
        conf = 0.5
    return {"label": lab, "quote": str(j.get("quote", "") or ""), "confidence": min(max(conf, 0.0), 1.0)}


def _label_from_prose(txt: str) -> str:
    """Last-resort reading of a non-JSON reply. Negated forms are checked first, so
    "NOT SUPPORTED" or "unsupported" is never read as SUPPORTED."""
    up = " ".join(txt.upper().split())
    if any(k in up for k in ("NOT SUPPORTED", "UNSUPPORTED", "NOT ENOUGH", "NEI", "INSUFFICIENT")):
        return "NEI"
    if any(k in up for k in ("REFUTED", "CONTRADICT", "FALSE")):
        return "REFUTED"
    if "SUPPORTED" in up:
        return "SUPPORTED"
    return "NEI"


def judgement_probs(j: dict) -> list[float]:
    """Pseudo-probabilities whose top label is ALWAYS the label the LLM stated.

    LLMs do not use "confidence" consistently: some answer {"label": "NEI",
    "confidence": 0.2}, meaning "probably not true" rather than "20% sure of NEI". An
    earlier version clamped such values to 1/3, which made a three-way tie, and the tie
    silently scored the claim as SUPPORTED. The stated label now gets at least 0.4 (the
    other two at most 0.3 each), so the argmax is the stated label; the confidence only
    shapes calibration metrics, which for LLM judges should be read with this caveat."""
    c = min(max(j["confidence"], 0.4), 1.0)
    return [c if l == j["label"] else (1 - c) / 2 for l in LABELS]
