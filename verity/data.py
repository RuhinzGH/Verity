"""Loaders for the three benchmarks, each mapped onto Verity's label set.

SciFact      claims about biomedical findings, checked against 5,183 paper abstracts
Climate-FEVER real-world climate claims, each with five Wikipedia evidence sentences
RAGTruth     answers written by six LLMs, with human-marked hallucinated spans

Run `python -m verity.data` once to download the raw files into data/raw/.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from .config import RAW

SOURCES = [
    ("allenai/scifact", "refs/convert/parquet", "claims/train/0000.parquet"),
    ("allenai/scifact", "refs/convert/parquet", "claims/validation/0000.parquet"),
    ("allenai/scifact", "refs/convert/parquet", "corpus/train/0000.parquet"),
    ("tdiggelm/climate_fever", None, "data/test-00000-of-00001.parquet"),
    ("wandb/RAGTruth-processed", None, "data/train-00000-of-00001.parquet"),
    ("wandb/RAGTruth-processed", None, "data/test-00000-of-00001.parquet"),
]


def _local(repo: str, path: str) -> str:
    return str(RAW / (repo.split("/")[1] + "__" + path.replace("/", "_")))


def download() -> None:
    """Fetch the plain parquet files (no dataset loading scripts are executed)."""
    from huggingface_hub import hf_hub_download
    RAW.mkdir(parents=True, exist_ok=True)
    for repo, rev, path in SOURCES:
        out = _local(repo, path)
        try:
            open(out).close()
            continue
        except FileNotFoundError:
            pass
        shutil.copy(hf_hub_download(repo, path, repo_type="dataset", revision=rev), out)
        print("downloaded", out)


# ------------------------------------------------------------------ data types
@dataclass
class Doc:
    doc_id: str
    title: str
    sentences: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass
class Claim:
    cid: str
    text: str
    label: str                                   # SUPPORTED / REFUTED / NEI
    evidence: dict[str, list[int]] = field(default_factory=dict)   # doc_id -> sentence ids
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------ SciFact
SCIFACT_MAP = {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED"}


@lru_cache(maxsize=None)
def scifact_corpus() -> dict[str, Doc]:
    df = pd.read_parquet(_local("allenai/scifact", "corpus/train/0000.parquet"))
    return {str(r.doc_id): Doc(str(r.doc_id), r.title, [str(s) for s in r.abstract])
            for r in df.itertuples()}


@lru_cache(maxsize=None)
def scifact_claims(split: str) -> list[Claim]:
    """split: 'train' or 'dev' (SciFact's public test split carries no labels)."""
    name = {"train": "train", "dev": "validation"}[split]
    df = pd.read_parquet(_local("allenai/scifact", f"claims/{name}/0000.parquet"))
    out: dict[int, Claim] = {}
    for r in df.itertuples():
        c = out.setdefault(r.id, Claim(f"sf-{r.id}", r.claim, "NEI",
                                       meta={"cited": [str(d) for d in r.cited_doc_ids]}))
        if r.evidence_label:
            c.evidence.setdefault(str(r.evidence_doc_id), []).extend(int(i) for i in r.evidence_sentences)
            c.meta.setdefault("doc_labels", {})[str(r.evidence_doc_id)] = SCIFACT_MAP[r.evidence_label]
    for c in out.values():
        labs = set(c.meta.get("doc_labels", {}).values())
        # a handful of claims are supported by one paper and contradicted by another;
        # they stay in retrieval experiments but are marked so label metrics can skip them
        if len(labs) == 1:
            c.label = labs.pop()
        elif len(labs) > 1:
            c.label = "MIXED"
    return list(out.values())


# ------------------------------------------------------------------ Climate-FEVER
CF_MAP = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}       # 3 = DISPUTED (dropped)


@lru_cache(maxsize=None)
def climate_fever() -> tuple[list[Claim], dict[str, Doc]]:
    """Claims plus a corpus pooled from every claim's evidence sentences (5,240 distinct)."""
    df = pd.read_parquet(_local("tdiggelm/climate_fever", "data/test-00000-of-00001.parquet"))
    claims, corpus = [], {}
    for r in df.itertuples():
        ev, pairs = {}, []
        for e in r.evidences:
            did = e["evidence_id"]
            corpus[did] = Doc(did, e["article"], [e["evidence"]])
            pairs.append((did, CF_MAP[int(e["evidence_label"])]))
            if int(e["evidence_label"]) in (0, 1):
                ev[did] = [0]
        if int(r.claim_label) == 3:
            continue
        claims.append(Claim(f"cf-{r.claim_id}", r.claim, CF_MAP[int(r.claim_label)], ev,
                            meta={"pairs": pairs}))
    return claims, corpus


# ------------------------------------------------------------------ RAGTruth
@dataclass
class Response:
    rid: str
    task: str
    model: str
    query: str
    context: str
    output: str
    spans: list[dict]              # hallucinated spans: start, end, text, kind

    @property
    def hallucinated(self) -> bool:
        return bool(self.spans)


@lru_cache(maxsize=None)
def ragtruth(split: str) -> list[Response]:
    df = pd.read_parquet(_local("wandb/RAGTruth-processed", f"data/{split}-00000-of-00001.parquet"))
    out = []
    for r in df.itertuples():
        spans = []
        for s in json.loads(r.hallucination_labels):
            kind = s.get("label_type") or s.get("meta", "").split(":")[0]
            spans.append({"start": s["start"], "end": s["end"], "text": s["text"], "kind": kind})
        out.append(Response(str(r.id), r.task_type, r.model, r.query, r.context, r.output, spans))
    return out


if __name__ == "__main__":
    download()
    print("SciFact corpus", len(scifact_corpus()), "| train", len(scifact_claims("train")),
          "| dev", len(scifact_claims("dev")))
    cf, cfc = climate_fever()
    print("Climate-FEVER claims", len(cf), "| evidence pool", len(cfc))
    print("RAGTruth train", len(ragtruth("train")), "| test", len(ragtruth("test")))
