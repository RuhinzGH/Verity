"""(evidence, claim, label) pairs for training and for the oracle verifier experiments.

SciFact, one pair per claim ("oracle evidence"):
  SUPPORTED / REFUTED  the annotated rationale sentences of the first evidence abstract
  NEI                  the sentences of a cited abstract most similar to the claim
                       (the paper the claim cites, which turns out not to settle it)

SciFact training also adds a hard negative per evidence claim: the most similar
sentences of the *same* abstract that are not rationales, labelled NEI. That teaches the
verifier that being on-topic is not the same as being evidence.

Length matching. Rationales are usually one or two sentences. If NEI evidence were
always three sentences, a model could tell NEI apart by length alone (an audit found
exactly this in the first version). So every NEI pair has as many sentences as a
rationale would: a hard negative takes the same count as its own claim's rationale, and
a cited-abstract NEI pair draws its count from the split's rationale lengths (seeded
per claim, so the pairs are reproducible).

Climate-FEVER: each of a claim's five evidence sentences with its own annotated label.
"""
from __future__ import annotations

import random

import numpy as np

from . import data
from .config import SEED


def top_sentences(claim: str, sents: list[str], k: int = 3, exclude: set[int] = frozenset()) -> list[int]:
    from .retrieve import embed
    idx = [i for i in range(len(sents)) if i not in exclude]
    if not idx:
        return []
    q = embed("bge-base", [claim], query=True, cache=False)[0]
    s = embed("bge-base", [sents[i] for i in idx], cache=False) @ q
    order = [idx[j] for j in np.argsort(-s)[:k]]
    return sorted(order)


def _rationale_lengths(split: str) -> list[int]:
    out = []
    for c in data.scifact_claims(split):
        if c.label in ("SUPPORTED", "REFUTED"):
            out.append(len(set(c.evidence[next(iter(c.evidence))])))
    return out


def scifact_pairs(split: str, hard_negatives: bool = False) -> list[dict]:
    corpus = data.scifact_corpus()
    lengths = _rationale_lengths(split)
    out = []
    for c in data.scifact_claims(split):
        if c.label in ("SUPPORTED", "REFUTED"):
            did = next(iter(c.evidence))
            sents = corpus[did].sentences
            rat = sorted(set(c.evidence[did]))
            out.append({"cid": c.cid, "claim": c.text, "evidence": " ".join(sents[i] for i in rat),
                        "label": c.label, "doc": did, "kind": "rationale"})
            if hard_negatives:
                neg = top_sentences(c.text, sents, len(rat), exclude=set(rat))
                if neg:
                    out.append({"cid": c.cid, "claim": c.text, "evidence": " ".join(sents[i] for i in neg),
                                "label": "NEI", "doc": did, "kind": "same-abstract non-rationale"})
        elif c.label == "NEI":
            cited = [d for d in c.meta.get("cited", []) if d in corpus]
            if not cited:
                continue
            sents = corpus[cited[0]].sentences
            k = random.Random(f"{SEED}-{c.cid}").choice(lengths)
            pick = top_sentences(c.text, sents, k)
            out.append({"cid": c.cid, "claim": c.text, "evidence": " ".join(sents[i] for i in pick),
                        "label": "NEI", "doc": cited[0], "kind": "cited abstract"})
    return out


def split_train_calibration(pairs: list[dict], cal_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Split by claim (never by pair) so no claim is in both halves."""
    cids = sorted({p["cid"] for p in pairs})
    rng = random.Random(SEED)
    rng.shuffle(cids)
    cal = set(cids[:int(len(cids) * cal_frac)])
    return [p for p in pairs if p["cid"] not in cal], [p for p in pairs if p["cid"] in cal]


def climate_pairs() -> list[dict]:
    claims, corpus = data.climate_fever()
    return [{"cid": c.cid, "claim": c.text, "evidence": corpus[did].text, "label": lab,
             "doc": did, "kind": "climate evidence"}
            for c in claims for did, lab in c.meta["pairs"]]
