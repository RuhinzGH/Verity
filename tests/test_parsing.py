"""decompose.parse and verify.parse_judgement / judgement_probs / softmax."""
import json

import numpy as np
import pytest

from verity import decompose
from verity.config import LABELS
from verity.verify import judgement_probs, parse_judgement, softmax


# ------------------------------------------------------------------ decompose.parse
def test_parse_valid_json():
    raw = json.dumps({"claims": [{"s": 1, "claim": "A is B."}, {"s": 2, "claim": " C is D. "}]})
    assert decompose.parse(raw, 2) == [(0, "A is B."), (1, "C is D.")]


@pytest.mark.parametrize("raw", [
    'Sure! Here are the claims:\n{"claims": [{"s": 1, "claim": "A is B."}]}\nHope this helps.',
    '```json\n{"claims": [{"s": 1, "claim": "A is B."}]}\n```',
    '```\n{"claims": [{"s": 1, "claim": "A is B."}]}```',
])
def test_parse_json_in_prose_or_fences(raw):
    assert decompose.parse(raw, 3) == [(0, "A is B.")]


@pytest.mark.parametrize("raw", [
    "", "no json here", '{"claims": [', '{"claim": "x"}', "[1, 2, 3]", '{"claims": [{"s": 1, }]}',
])
def test_parse_malformed_returns_none(raw):
    assert decompose.parse(raw, 3) is None


def test_parse_sentence_index_clamped():
    raw = json.dumps({"claims": [{"s": 0, "claim": "low"}, {"s": -4, "claim": "neg"},
                                 {"s": 99, "claim": "high"}, {"s": "2", "claim": "str"}]})
    assert decompose.parse(raw, 3) == [(0, "low"), (0, "neg"), (2, "high"), (1, "str")]


def test_parse_skips_bad_items():
    raw = json.dumps({"claims": [{"s": 1}, "just a string", {"s": "x", "claim": "bad s"},
                                 {"s": 2, "claim": "   "}, {"claim": "no s defaults to 1"}, 7,
                                 {"s": None, "claim": "null s"}, {"s": 3, "claim": "ok"}]})
    assert decompose.parse(raw, 3) == [(0, "no s defaults to 1"), (2, "ok")]


@pytest.mark.parametrize("raw", ['{"claims": null}', '{"claims": 5}'])
def test_parse_non_list_claims_falls_back(raw):
    assert decompose.parse(raw, 3) is None


def test_parse_string_claims_falls_back():
    assert decompose.parse('{"claims": "A is B."}', 3) is None


def test_by_llm_falls_back_to_sentences_on_garbage(monkeypatch):
    class FakeLLM:
        def __init__(self, model):
            pass

        def complete(self, prompt, max_tokens=800):
            return "I cannot do that."
    import verity.llm
    monkeypatch.setattr(verity.llm, "LLM", FakeLLM)
    answer = "Paris is in France. It is big."
    out = decompose.by_llm(answer, "fake")
    assert [c.text for c in out] == ["Paris is in France.", "It is big."]


def test_by_llm_maps_claims_to_sentence_spans(monkeypatch):
    class FakeLLM:
        def __init__(self, model):
            pass

        def complete(self, prompt, max_tokens=800):
            assert "[1] Paris is in France." in prompt and "[2] It is big." in prompt
            return json.dumps({"claims": [{"s": 1, "claim": "Paris is in France."},
                                          {"s": 2, "claim": "Paris is big."}]})
    import verity.llm
    monkeypatch.setattr(verity.llm, "LLM", FakeLLM)
    answer = "Paris is in France. It is big."
    out = decompose.by_llm(answer, "fake")
    assert [(c.text, c.sentence) for c in out] == [("Paris is in France.", 0), ("Paris is big.", 1)]
    assert answer[out[1].start:out[1].end] == "It is big."


# ------------------------------------------------------------------ verify.parse_judgement
def test_judgement_valid():
    j = parse_judgement('{"label": "REFUTED", "quote": "no effect", "confidence": 0.8}')
    assert j == {"label": "REFUTED", "quote": "no effect", "confidence": 0.8}


@pytest.mark.parametrize("raw", [
    'Reasoning first.\n{"label": "SUPPORTED", "quote": "q", "confidence": 0.9}\nDone.',
    '```json\n{"label": "SUPPORTED", "quote": "q", "confidence": 0.9}\n```',
    '  {"label": "supported", "quote": "q", "confidence": "0.9"}  ',
    '{"label": "entailment", "quote": "q", "confidence": 0.9}',
])
def test_judgement_wrapped_and_aliases(raw):
    j = parse_judgement(raw)
    assert j["label"] == "SUPPORTED"
    assert j["confidence"] == pytest.approx(0.9)
    assert "parse_error" not in j


@pytest.mark.parametrize("label,expected", [
    ("contradiction", "REFUTED"), ("neutral", "NEI"), ("not_entailment", "NEI"), ("NEI", "NEI"),
    ("MAYBE", "NEI"), ("", "NEI"), (None, "NEI"), (3, "NEI"), ("Refuted ", "REFUTED"),
])
def test_judgement_label_mapping(label, expected):
    assert parse_judgement(json.dumps({"label": label, "confidence": 0.7}))["label"] == expected


def test_judgement_missing_fields_defaults():
    j = parse_judgement("{}")
    assert j == {"label": "NEI", "quote": "", "confidence": 0.5}
    assert parse_judgement('{"label": "REFUTED", "quote": null}')["quote"] == ""


@pytest.mark.parametrize("conf,expected", [
    (1.7, 1.0), (-0.2, 0.0), (85, 1.0), ("high", 0.5), (None, 0.5), ([0.3], 0.5), (0.25, 0.25),
])
def test_judgement_confidence_clamped(conf, expected):
    j = parse_judgement(json.dumps({"label": "SUPPORTED", "confidence": conf}))
    assert j["confidence"] == pytest.approx(expected)
    assert 0.0 <= j["confidence"] <= 1.0


def test_judgement_nan_confidence_clamped():
    j = parse_judgement('{"label": "SUPPORTED", "confidence": NaN}')
    assert 0.0 <= j["confidence"] <= 1.0
    assert np.isfinite(judgement_probs(j)).all()


@pytest.mark.parametrize("raw,label", [
    ("The answer is REFUTED.", "REFUTED"),
    ('{"label": "REFUTED", "confidence": ', "REFUTED"),
    ("SUPPORTED", "SUPPORTED"),
    ("I am not sure.", "NEI"),
    ("", "NEI"),
])
def test_judgement_malformed_fallback(raw, label):
    j = parse_judgement(raw)
    assert j["label"] == label
    assert j["confidence"] == 0.5
    assert j["parse_error"] is True


@pytest.mark.parametrize("raw", ["Label: NOT SUPPORTED", "This claim is UNSUPPORTED by the evidence."])
def test_judgement_fallback_negated_label(raw):
    assert parse_judgement(raw)["label"] != "SUPPORTED"


# ------------------------------------------------------------------ judgement_probs / softmax
@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("conf", [0.0, 0.2, 1 / 3, 0.5, 0.9, 1.0])
def test_judgement_probs_sum_to_one(label, conf):
    p = judgement_probs({"label": label, "confidence": conf})
    assert sum(p) == pytest.approx(1.0)
    assert all(0 <= x <= 1 for x in p)
    i = LABELS.index(label)
    assert p[i] == pytest.approx(min(max(conf, 0.4), 1.0))   # stated label always on top
    assert p[i] == pytest.approx(max(p))


def test_judgement_probs_values():
    assert judgement_probs({"label": "REFUTED", "confidence": 0.8}) == pytest.approx([0.1, 0.8, 0.1])


def test_softmax():
    z = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    p = softmax(z)
    assert np.allclose(p.sum(1), 1)
    assert np.isfinite(p).all()
    assert np.allclose(p[0], 1 / 3)
    assert p[1, 0] == pytest.approx(1.0)
    e = np.exp([1.0, 2.0, 3.0])
    assert np.allclose(p[2], e / e.sum())
    # higher temperature flattens, ranking unchanged
    hot = softmax(z[2:], t=10.0)[0]
    assert hot.max() < p[2].max()
    assert np.argsort(hot).tolist() == np.argsort(p[2]).tolist()
    assert np.allclose(softmax(z[2:] * 2, t=2.0), p[2:])


@pytest.mark.parametrize("label", ["SUPPORTED", "REFUTED", "NEI"])
@pytest.mark.parametrize("conf", [0.0, 0.1, 0.2, 1 / 3, 0.5, 1.0])
def test_judgement_probs_top_label_is_the_stated_label(label, conf):
    # regression: low stated confidence used to tie all three labels, and argmax then said SUPPORTED
    import numpy as np
    from verity.config import LABELS
    p = judgement_probs({"label": label, "confidence": conf})
    assert LABELS[int(np.argmax(p))] == label
    assert abs(sum(p) - 1) < 1e-9
