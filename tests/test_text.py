import pytest

from verity.text import sentence_spans, sentences, windows


def check_offsets(text):
    spans = sentence_spans(text)
    for s, a, e in spans:
        assert 0 <= a <= e <= len(text)
        assert text[a:e].strip() == s
    starts = [a for _, a, _ in spans]
    assert starts == sorted(starts)
    for (_, _, e1), (_, a2, _) in zip(spans, spans[1:]):
        assert e1 <= a2, "spans must not overlap"
    return spans


@pytest.mark.parametrize("text", [
    "The cat sat. The dog ran! Did it? Yes.",
    "  Leading spaces. And trailing ones.   ",
    'He said "Hi." Then he left.',
    "Results (see Fig. 3) were clear. In 2020, rates fell. 42 people agreed.",
    "Line one\nLine two. Still two.\n\n\nLine three",
    "Windows line.\r\nNext line. Done.\r\n",
    "Wait... Then it happened.",
    "",
    "x",
])
def test_offsets_map_back_onto_text(text):
    check_offsets(text)


def test_basic_split():
    assert sentences("The cat sat. The dog ran! Did it? Yes.") == \
        ["The cat sat.", "The dog ran!", "Did it?", "Yes."]


@pytest.mark.parametrize("text", [
    "Use a tool, e.g. Python is popular.",
    "That is, i.e. The main point stands.",
    "Dr. Smith wrote it.",
    "Mr. and Mrs. Jones came.",
    "He moved to the U.S. Army base in May.",
    "J. K. Rowling wrote the book.",
    "Smith et al. Reported a gain.",
    "See Fig. 2 for details.",
])
def test_abbreviations_and_initials_do_not_split(text):
    spans = check_offsets(text)
    assert len(spans) == 1, spans
    assert spans[0][0] == text


def test_mixed_abbreviations_still_split_real_boundaries():
    text = "Dr. Smith went to the U.S. Army base. He met J. K. Rowling there, e.g. Monday. Then left."
    spans = check_offsets(text)
    assert [s for s, _, _ in spans] == [
        "Dr. Smith went to the U.S. Army base.",
        "He met J. K. Rowling there, e.g. Monday.",
        "Then left.",
    ]


def test_closing_quote_and_bracket_stay_with_sentence():
    assert sentences('He said "Hi." Then he left.') == ['He said "Hi."', "Then he left."]
    assert sentences("It worked (mostly.) Then it broke.") == ["It worked (mostly.)", "Then it broke."]


def test_lowercase_after_period_does_not_split():
    assert len(sentences("The value was 3.5 percent. the rest followed.")) == 1


def test_newlines_split_even_without_punctuation():
    text = "First line without stop\nSecond line\r\nThird."
    spans = check_offsets(text)
    assert [s for s, _, _ in spans] == ["First line without stop", "Second line", "Third."]


def test_single_character_fragments_dropped():
    assert sentences("x") == []
    assert sentences("A real sentence.\nx\n") == ["A real sentence."]


def test_offsets_exact_without_strip():
    text = "  Hello world. Next one."
    for s, a, e in sentence_spans(text):
        assert text[a:e] == s


def test_windows():
    s = ["a", "b", "c", "d"]
    assert windows(s, 2) == ["a b", "b c", "c d"]
    assert windows(s, 3) == ["a b c", "b c d"]
    assert windows(s, 4) == ["a b c d"]
    assert windows(s, 9) == ["a b c d"]
    assert windows(["only"], 2) == ["only"]
    assert windows([], 2) == []
    assert windows(s, 1) == s
