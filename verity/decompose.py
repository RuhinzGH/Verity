"""Claim decomposition: turn an answer into short claims that can each be checked.

  sentences   baseline: every sentence is one claim
  llm         an LLM rewrites each sentence into atomic, self-contained claims
              (one fact each, pronouns replaced by what they refer to), and says which
              sentence each claim came from, so a verdict can be mapped back onto the text
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .text import sentence_spans

PROMPT = """Break the ANSWER below into atomic claims for fact-checking.

Rules:
- One fact per claim. Split sentences that state several facts.
- Each claim must make sense on its own: replace pronouns and vague references
  ("it", "they", "the company") with what they refer to.
- Keep the answer's own wording and numbers; do not add facts or fix errors.
- Skip text that states no checkable fact (greetings, opinions, "I hope this helps").
- "s" is the number of the sentence the claim comes from.

ANSWER (sentences numbered):
{numbered}

Reply with JSON only: {{"claims": [{{"s": 1, "claim": "..."}}, ...]}}"""


@dataclass
class ClaimSpan:
    text: str              # the claim to check
    sentence: int          # index of the source sentence
    start: int             # character span of that sentence in the answer
    end: int


def by_sentence(answer: str) -> list[ClaimSpan]:
    return [ClaimSpan(s, i, a, e) for i, (s, a, e) in enumerate(sentence_spans(answer))]


def by_llm(answer: str, model: str) -> list[ClaimSpan]:
    from .llm import LLM
    spans = sentence_spans(answer)
    if not spans:
        return []
    numbered = "\n".join(f"[{i + 1}] {s}" for i, (s, _, _) in enumerate(spans))
    raw = LLM(model).complete(PROMPT.format(numbered=numbered), max_tokens=1500)
    claims = parse(raw, len(spans))
    if claims is None:                       # unparseable reply: fall back to sentences
        return by_sentence(answer)
    return [ClaimSpan(t, i, spans[i][1], spans[i][2]) for i, t in claims]


def prefetch(answers: list[str], model: str) -> None:
    """Generate (and cache) LLM decompositions for many answers in GPU batches;
    by_llm() then reads them from the cache."""
    from .llm import LLM
    prompts = []
    for a in answers:
        spans = sentence_spans(a)
        if spans:
            numbered = "\n".join(f"[{i + 1}] {s}" for i, (s, _, _) in enumerate(spans))
            prompts.append(PROMPT.format(numbered=numbered))
    LLM(model).complete_many(prompts, max_tokens=1500, batch_size=6, progress=f"decompose {model}")


def parse(raw: str, n_sentences: int) -> list[tuple[int, str]] | None:
    s, e = raw.find("{"), raw.rfind("}")
    try:
        items = json.loads(raw[s:e + 1])["claims"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(items, list):          # null, a number or a bare string: unusable
        return None
    out = []
    for it in items:
        try:
            i = int(it.get("s", 1)) - 1
            t = str(it["claim"]).strip()
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if t:
            out.append((min(max(i, 0), n_sentences - 1), t))
    return out
