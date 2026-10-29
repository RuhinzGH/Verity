"""Small text helpers: sentence splitting with character offsets."""
from __future__ import annotations

import re

# A sentence ends at . ! ? (optionally followed by a quote or bracket) before whitespace
# and an upper-case letter, digit or opening quote ...
_BOUNDARY = re.compile(r"[.!?][\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
# ... unless the full stop belongs to an abbreviation or a single initial.
_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|St|No|Fig|al|approx|U\.S)|\b[A-Z])\.$")


def sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """(sentence, start, end) with offsets into `text`; a newline also ends a sentence."""
    out = []
    for block in re.finditer(r"[^\n]+", text):
        b, base, start = block.group(), block.start(), 0
        for m in _BOUNDARY.finditer(b):
            piece = b[start:m.start() + 1]
            if _ABBREV.search(piece):
                continue
            end = m.end() - len(m.group()) + len(m.group().rstrip())
            out.append((b[start:end].strip(), base + start, base + end))
            start = m.end()
        if b[start:].strip():
            out.append((b[start:].strip(), base + start, base + len(b.rstrip())))
    # move each start past leading whitespace so text[start:end] is exactly the sentence
    fixed = []
    for s, a, e in out:
        while a < e and text[a].isspace():
            a += 1
        fixed.append((s, a, e))
    return [(s, a, e) for s, a, e in fixed if len(s) > 1]


def sentences(text: str) -> list[str]:
    return [s for s, _, _ in sentence_spans(text)]


def windows(sents: list[str], size: int = 2) -> list[str]:
    """Overlapping windows of `size` sentences (passages for retrieval inside a context)."""
    if len(sents) <= size:
        return [" ".join(sents)] if sents else []
    return [" ".join(sents[i:i + size]) for i in range(len(sents) - size + 1)]
