"""Benchmark loaders; skipped unless the raw parquet files are present in data/raw/."""
import pytest

from verity import data
from verity.config import LABELS, RAW


def need(*names):
    missing = [n for n in names if not (RAW / n).exists()]
    return pytest.mark.skipif(bool(missing), reason=f"raw data missing: {missing}")


SF_DEV = "scifact__claims_validation_0000.parquet"
SF_TRAIN = "scifact__claims_train_0000.parquet"
SF_CORPUS = "scifact__corpus_train_0000.parquet"
CF = "climate_fever__data_test-00000-of-00001.parquet"
RT_TEST = "RAGTruth-processed__data_test-00000-of-00001.parquet"


@need(SF_CORPUS)
def test_scifact_corpus():
    corpus = data.scifact_corpus()
    assert len(corpus) == 5183
    d = next(iter(corpus.values()))
    assert d.sentences and all(isinstance(s, str) for s in d.sentences)
    assert d.text == " ".join(d.sentences)


@need(SF_DEV, SF_CORPUS)
def test_scifact_dev():
    claims = data.scifact_claims("dev")
    assert len(claims) == 300
    assert len({c.cid for c in claims}) == 300
    assert {c.label for c in claims} <= set(LABELS) | {"MIXED"}
    corpus = data.scifact_corpus()
    for c in claims:
        if c.label in ("SUPPORTED", "REFUTED"):
            assert c.evidence
            for did, sids in c.evidence.items():
                assert did in corpus
                assert all(0 <= i < len(corpus[did].sentences) for i in sids)
        if c.label == "NEI":
            assert not c.evidence


@need(SF_TRAIN)
def test_scifact_train_labels():
    claims = data.scifact_claims("train")
    assert len(claims) > 0
    assert {c.label for c in claims} <= set(LABELS) | {"MIXED"}


@need(CF)
def test_climate_fever():
    claims, corpus = data.climate_fever()
    assert len(claims) == 1381
    assert {c.label for c in claims} <= set(LABELS)
    for c in claims:
        assert len(c.meta["pairs"]) == 5
        assert all(did in corpus for did, _ in c.meta["pairs"])
        assert set(c.evidence) <= {did for did, _ in c.meta["pairs"]}


@need(RT_TEST)
def test_ragtruth_test():
    rs = data.ragtruth("test")
    assert len(rs) == 2700
    for r in rs[:300]:
        assert r.hallucinated == bool(r.spans)
        for s in r.spans:
            assert 0 <= s["start"] <= s["end"] <= len(r.output)
