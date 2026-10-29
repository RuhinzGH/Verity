"""End-to-end Verity.check with the NLI verifier and the dense encoder replaced by cheap fakes.

Real BM25 (bm25s, CPU) still runs; no model is downloaded or loaded.
"""
import hashlib
import json
import re

import numpy as np
import pytest

from verity import pipeline, retrieve
from verity.config import LABELS

ALLOWED = set(LABELS) | {"UNCERTAIN"}


def test_calibration_is_per_verifier(tmp_path, monkeypatch):
    from verity.config import FINETUNED_NLI
    r = {"models": {"verity-finetuned": {"temperature": 2.0, "tests": {"scifact_dev": {"conformal": {"0.1": {"qhat": 0.7}}}}},
                    "deberta-fever": {"temperature": 3.0, "tests": {"scifact_dev": {"conformal": {"0.1": {"qhat": 0.8}}}}}},
         "deployed": {"temperature": 2.0, "qhat": 0.7}}
    (tmp_path / "e04_conformal.json").write_text(json.dumps(r))
    monkeypatch.setattr(pipeline, "RESULTS", tmp_path)
    assert pipeline._load_calibration(str(FINETUNED_NLI)) == (2.0, 0.7)
    assert pipeline._load_calibration("deberta-fever") == (3.0, 0.8)
    assert pipeline._load_calibration("something-else") == (2.0, 0.7)
    monkeypatch.setattr(pipeline, "RESULTS", tmp_path / "missing")
    assert pipeline._load_calibration("deberta-fever") == (1.0, 0.9)


def words(t):
    return set(re.findall(r"[a-z0-9]+", t.lower())) - {"the", "a", "is", "of", "in", "and", "was"}


def fake_embed(name, texts, query=False, cache=True, dim=64):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for r, t in enumerate(texts):
        for w in words(t):
            out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out


class FakeNLI:
    """SUPPORTED logit grows with word overlap; 'never' in the claim flips it to REFUTED."""

    def __init__(self):
        self.calls = []

    def logits(self, pairs):
        self.calls.append(list(pairs))
        out = []
        for ev, cl in pairs:
            cw = words(cl)
            f = len(cw & words(ev)) / max(len(cw), 1)
            if "never" in cl.lower():
                out.append([0.0, 8.0 * f, 2.0])
            else:
                out.append([8.0 * f, 0.0, 2.0])
        return np.array(out, dtype=np.float32)


@pytest.fixture
def fakes(monkeypatch):
    nli = FakeNLI()
    names = []

    def fake_nli(name):
        names.append(name)
        return nli
    monkeypatch.setattr(pipeline, "nli", fake_nli)
    monkeypatch.setattr(retrieve, "embed", fake_embed)
    monkeypatch.setattr(retrieve, "_encoder", lambda name: pytest.fail("real encoder loaded"))
    monkeypatch.setattr(pipeline, "_load_calibration", lambda *a: (1.0, 0.5))
    return nli, names


SOURCES = [
    ("paris", "Paris is the capital of France. The Eiffel Tower stands in Paris. "
              "It was completed in 1889."),
    ("bio", "Insulin is produced by the pancreas. Insulin lowers blood glucose. "
            "Type 1 diabetes destroys beta cells."),
    ("misc", "Penguins live in the Southern Hemisphere. Penguins cannot fly."),
]

ANSWER = ("The Eiffel Tower stands in Paris.\n"
          "  Insulin is never produced by the pancreas. "
          "Quantum gravity remains unsolved by physicists.")


def test_check_end_to_end(fakes):
    nli, names = fakes
    v = pipeline.Verity(sources=SOURCES, top_k=3, window=2)
    # passages are sliding windows over each source, with ids name#j
    assert [p for p, _ in v.passages] == ["paris#0", "paris#1", "bio#0", "bio#1", "misc#0"]
    assert v.origin["bio#1"] == "bio"

    rep = v.check(ANSWER)
    assert len(rep.claims) == 3
    for c in rep.claims:
        assert ANSWER[c.start:c.end].strip() == c.claim
        assert c.verdict in ALLOWED
        assert set(c.label_set) <= set(LABELS)
        assert list(c.probs) == LABELS
        assert sum(c.probs.values()) == pytest.approx(1.0, abs=1e-5)
        assert len(c.evidence) == 3
        keys = [max(e.probs["SUPPORTED"], e.probs["REFUTED"]) for e in c.evidence]
        assert keys == sorted(keys, reverse=True), "evidence must be ordered by decisiveness"
        assert c.evidence[0].probs == pytest.approx(c.probs), "verdict comes from the top passage"
        assert all(e.source in {"paris", "bio", "misc"} for e in c.evidence)
        assert all(e.text in v.text.values() for e in c.evidence)
    assert [c.sentence for c in rep.claims] == [0, 1, 2]

    sup, ref, nei = rep.claims
    assert sup.verdict == "SUPPORTED" and sup.evidence[0].source == "paris"
    assert ref.verdict == "REFUTED" and ref.evidence[0].source == "bio"
    assert nei.verdict == "NEI"

    # one NLI call, one pair per (claim, candidate), passed as (evidence, claim)
    assert len(nli.calls) == 1 and len(nli.calls[0]) == 9
    assert all(ev in v.text.values() for ev, _ in nli.calls[0])
    assert names == [v.verifier]

    assert rep.counts == {"SUPPORTED": 1, "REFUTED": 1, "NEI": 1, "UNCERTAIN": 0}
    j = json.loads(rep.to_json())
    assert j["answer"] == ANSWER
    assert j["counts"] == rep.counts
    assert [c["verdict"] for c in j["claims"]] == ["SUPPORTED", "REFUTED", "NEI"]
    assert set(j["claims"][0]["evidence"][0]) == {"source", "text", "probs"}


def test_uncertain_with_loose_threshold(fakes, monkeypatch):
    # qhat = 1 keeps every label in the set -> UNCERTAIN
    monkeypatch.setattr(pipeline, "_load_calibration", lambda *a: (1.0, 1.0))
    rep = pipeline.Verity(sources=SOURCES).check("Penguins cannot fly.")
    assert [c.verdict for c in rep.claims] == ["UNCERTAIN"]
    assert rep.claims[0].label_set == LABELS
    assert rep.counts["UNCERTAIN"] == 1


def test_top_k_larger_than_corpus(fakes):
    v = pipeline.Verity(sources=SOURCES, top_k=50)
    rep = v.check("Penguins cannot fly.")
    assert len(rep.claims[0].evidence) == len(v.passages)
    assert rep.claims[0].evidence[0].source == "misc"


def test_single_passage_skips_retrieval(fakes):
    v = pipeline.Verity(sources=[("one", "Penguins cannot fly.")])
    assert v.retriever is None
    rep = v.check("Penguins cannot fly. Penguins live in caves.")
    assert len(rep.claims) == 2
    assert all(len(c.evidence) == 1 and c.evidence[0].source == "one" for c in rep.claims)
    assert rep.claims[0].verdict == "SUPPORTED"


def test_empty_answer_and_empty_sources(fakes):
    v = pipeline.Verity(sources=SOURCES)
    rep = v.check("   \n ")
    assert rep.claims == [] and sum(rep.counts.values()) == 0
    assert json.loads(rep.to_json())["claims"] == []
    with pytest.raises(ValueError):
        pipeline.Verity(sources=[("empty", "")])


def test_no_split_sources(fakes):
    v = pipeline.Verity(sources=SOURCES, split_sources=False)
    assert [p for p, _ in v.passages] == ["paris#0", "bio#0", "misc#0"]
    rep = v.check("Insulin lowers blood glucose.")
    assert rep.claims[0].verdict == "SUPPORTED"
    assert rep.claims[0].evidence[0].source == "bio"
