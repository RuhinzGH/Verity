"""The whole of Verity in one call: answer in, checked claims out.

    from verity.pipeline import Verity
    v = Verity(sources=[("doc1", "text ..."), ...])        # or Verity.scifact()
    report = v.check("An AI-written answer ...")

Steps for each claim:
  1. decompose  split the answer into claims (sentences, or an LLM)
  2. retrieve   find the passages most likely to settle it (hybrid BM25 + embeddings):
                2-sentence windows of pasted sources, or the top 3 SciFact abstracts
  3. window     the verifier reads 2-sentence windows (an abstract is cut into all of its
                windows; a pasted-source passage already is one)
  4. verify     NLI verifier scores (evidence, claim) for every window
  5. decide     the window with the strongest SUPPORTED or REFUTED probability wins (chosen
                on uncalibrated probabilities, as in experiment 5b); conformal prediction
                turns its calibrated probabilities into a verdict or UNCERTAIN
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, dataclass, field

import numpy as np

from . import conformal, decompose, retrieve
from .config import FINETUNED_NLI, LABELS, LLM_DECOMPOSER, RESULTS
from .text import sentences, windows
from .verify import nli, softmax


@dataclass
class Evidence:
    source: str
    text: str
    probs: dict


@dataclass
class CheckedClaim:
    claim: str
    sentence: int
    start: int
    end: int
    verdict: str                       # SUPPORTED / REFUTED / NEI / UNCERTAIN
    label_set: list[str]
    probs: dict
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Report:
    answer: str
    claims: list[CheckedClaim]

    @property
    def counts(self) -> dict:
        out = {k: 0 for k in LABELS + ["UNCERTAIN"]}
        for c in self.claims:
            out[c.verdict] += 1
        return out

    def to_json(self) -> str:
        return json.dumps({"answer": self.answer, "counts": self.counts,
                           "claims": [asdict(c) for c in self.claims]}, indent=1)


def _load_calibration(verifier: str = "") -> tuple[float, float]:
    """Temperature and 90% conformal threshold that experiment 4 fitted for this verifier
    on the held-out calibration claims (falls back to the deployed model, then defaults)."""
    p = RESULTS / "e04_conformal.json"
    if not p.exists():
        return 1.0, 0.9
    r = json.loads(p.read_text())
    name = "verity-finetuned" if Path(verifier).name == FINETUNED_NLI.name else verifier
    m = r["models"].get(name)
    if m:
        return m["temperature"], m["tests"]["scifact_dev"]["conformal"]["0.1"]["qhat"]
    return r["deployed"]["temperature"], r["deployed"]["qhat"]


class Verity:
    def __init__(self, sources: list[tuple[str, str]], verifier: str | None = None,
                 decomposer: str = "sentences", top_k: int = 3, window: int = 2,
                 split_sources: bool = True, select: int | str | None = None):
        """sources: (name, text) documents to check against. Long documents are cut into
        overlapping windows of `window` sentences so evidence stays specific.

        select: how a retrieved passage is cut before verification. None = read it whole;
        "windows" = every 2-sentence window (experiment 5b's chosen design for whole
        abstracts, used by Verity.scifact); an int k = its k sentences closest to the claim."""
        self.passages, self.origin = [], {}
        for name, text in sources:
            chunks = windows(sentences(text), window) if split_sources else [text]
            for j, ch in enumerate(chunks):
                pid = f"{name}#{j}"
                self.passages.append((pid, ch))
                self.origin[pid] = name
        if not self.passages:
            raise ValueError("No source text to check against.")
        self.retriever = retrieve.Hybrid(self.passages) if len(self.passages) > 1 else None
        self.text = dict(self.passages)
        self.verifier = verifier or (str(FINETUNED_NLI) if FINETUNED_NLI.exists() else "deberta-fever")
        self.decomposer, self.top_k, self.select = decomposer, top_k, select
        self.temperature, self.qhat = _load_calibration(self.verifier)

    @classmethod
    def scifact(cls, **kw) -> "Verity":
        """Search all 5,183 SciFact abstracts. Uses the design experiment 5b chose on held-out
        claims: find the 3 most relevant abstracts, then let the fine-tuned verifier read
        every 2-sentence window inside them."""
        from .data import scifact_corpus
        docs = [(d.doc_id, f"{d.title}. {d.text}") for d in scifact_corpus().values()]
        kw.setdefault("select", "windows")
        return cls(docs, split_sources=False, **kw)

    def _units(self, pid: str, claim: str) -> list[str]:
        """The pieces of a retrieved passage the verifier reads (the strongest one counts)."""
        if not self.select:
            return [self.text[pid]]
        sents = sentences(self.text[pid])
        if self.select == "windows":
            return windows(sents, 2) or [self.text[pid]]
        from .pairs import top_sentences
        return [" ".join(sents[i] for i in top_sentences(claim, sents, int(self.select)))]

    # -------------------------------------------------------------- steps
    def claims(self, answer: str) -> list[decompose.ClaimSpan]:
        if self.decomposer == "sentences":
            return decompose.by_sentence(answer)
        return decompose.by_llm(answer, LLM_DECOMPOSER if self.decomposer == "llm" else self.decomposer)

    def candidates(self, claims: list[str]) -> list[list[str]]:
        if self.retriever is None:
            return [[self.passages[0][0]] for _ in claims]
        return [[pid for pid, _ in r] for r in self.retriever.search(claims, self.top_k)]

    def check(self, answer: str) -> Report:
        cs = self.claims(answer)
        if not cs:
            return Report(answer, [])
        cands = self.candidates([c.text for c in cs])
        units = [[self._units(pid, c.text) for pid in cand] for c, cand in zip(cs, cands)]
        pairs = [(u, c.text) for c, us in zip(cs, units) for per_pid in us for u in per_pid]
        z = nli(self.verifier).logits(pairs)
        raw, probs = softmax(z), softmax(z, self.temperature)
        out, i = [], 0
        for c, cand, us in zip(cs, cands, units):
            per = []                                 # strongest unit of each candidate passage
            for pid, per_pid in zip(cand, us):
                q, qr = probs[i:i + len(per_pid)], raw[i:i + len(per_pid)]
                i += len(per_pid)
                j = int(np.argmax(np.maximum(qr[:, 0], qr[:, 1])))
                per.append((pid, per_pid[j], q[j], max(qr[j, 0], qr[j, 1])))
            # the decisive window is chosen on uncalibrated probabilities, exactly as in
            # experiment 5b; the verdict then uses the calibrated ones
            per.sort(key=lambda t: -t[3])
            v = conformal.verdicts(per[0][2][None, :], self.qhat)[0]
            ev = [Evidence(self.origin[pid], text, {l: float(x) for l, x in zip(LABELS, q)})
                  for pid, text, q, _ in per]
            out.append(CheckedClaim(c.text, c.sentence, c.start, c.end, v["verdict"], v["set"],
                                    v["probs"], ev))
        return Report(answer, out)
