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


# ------------------------------------------------------------------ word overlap
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be been before being below
between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just may me might more
most much must my no nor not now of off on once only or other our ours out over own same she
should so some such than that the their theirs them then there these they this those through
to too under until up upon us very was we were what when where which while who whom why will
with would you your yours yet via per its one""".split())
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def stem(w: str) -> str:
    """Lower-case and strip a few English endings, so 'moons' meets 'moon'."""
    w = w.lower()
    if w.endswith("'s"):
        w = w[:-2]
    for suf, keep in (("ing", 4), ("ies", 3), ("ed", 3), ("es", 3), ("s", 3)):
        if w.endswith(suf) and len(w) - len(suf) >= keep and not w.endswith("ss"):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_words(text: str) -> set[str]:
    return {stem(m.group()) for m in WORD.finditer(text) if m.group().lower() not in STOPWORDS}


# a sentence that states several facts usually joins them at a comma, a semicolon or one of these
_PART = re.compile(r"[,;:]\s+|\s+(?=(?:and|but|while|whereas|although|though|yet|or|which|who)\b)", re.I)


def claim_parts(sentence: str, claims: list[str]) -> list[tuple[int, int, int]]:
    """Which part of a sentence each of its claims came from, so a sentence the LLM split
    into several claims can be coloured part by part.

    The sentence is cut at commas, semicolons and joining words. Each part goes to the
    claim whose own words (those no other claim of the sentence uses) it contains most; a
    part with none, or a tie, joins its neighbour. Returns (start, end, claim index)
    covering the whole sentence, or [] when fewer than two claims can be placed."""
    if len(claims) < 2:
        return []
    words = [content_words(c) for c in claims]
    own = [w - set().union(*(o for k, o in enumerate(words) if k != j)) for j, w in enumerate(words)]
    cuts = [0] + [m.end() for m in _PART.finditer(sentence)] + [len(sentence)]
    parts = []
    for a, e in zip(cuts, cuts[1:]):
        if a >= e:
            continue
        hits = [len(content_words(sentence[a:e]) & o) for o in own]
        best = max(hits)
        parts.append([a, e, hits.index(best) if best and hits.count(best) == 1 else None])
    for p, prev in zip(parts[1:], parts):          # an unplaced part joins the one before it
        if p[2] is None:
            p[2] = prev[2]
    first = next((p[2] for p in parts if p[2] is not None), None)
    merged: list[list] = []
    for p in parts:
        p[2] = first if p[2] is None else p[2]     # ... or, at the start, the first placed one
        if merged and merged[-1][2] == p[2]:
            merged[-1][1] = p[1]
        else:
            merged.append(p)
    if len({p[2] for p in merged}) < 2:
        return []
    return [(a, e, j) for a, e, j in merged]
