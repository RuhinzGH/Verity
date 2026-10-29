"""Evidence retrieval: four ways to find the passages a claim should be checked against.

    bm25     keyword matching (BM25 over stemmed words)
    dense    embedding similarity (bge-small or bge-base sentence embeddings)
    hybrid   reciprocal-rank fusion of bm25 and dense
    rerank   hybrid's top candidates re-scored by a cross-encoder that reads claim
             and passage together

Every retriever takes a list of (id, text) passages and returns, per query, a ranked
list of (id, score).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from .config import BGE_QUERY_PREFIX, BM25_STEMMER, CACHE, DENSE_MODELS, RERANKER

Ranking = list[tuple[str, float]]


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ BM25
class BM25:
    name = "bm25"

    def __init__(self, passages: list[tuple[str, str]]):
        import bm25s
        import Stemmer
        self.ids = [p[0] for p in passages]
        self.stemmer = Stemmer.Stemmer(BM25_STEMMER)
        self.index = bm25s.BM25()
        self.index.index(bm25s.tokenize([p[1] for p in passages], stopwords="en",
                                        stemmer=self.stemmer, show_progress=False),
                         show_progress=False)

    def search(self, queries: list[str], k: int = 10) -> list[Ranking]:
        import bm25s
        k = min(k, len(self.ids))
        toks = bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, show_progress=False)
        docs, scores = self.index.retrieve(toks, k=k, show_progress=False)
        return [[(self.ids[d], float(s)) for d, s in zip(dr, sr)] for dr, sr in zip(docs, scores)]


# ------------------------------------------------------------------ dense
@lru_cache(maxsize=None)
def _encoder(name: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(DENSE_MODELS[name], device=_device())
    if _device() == "cuda":
        m.half()
    return m


def embed(name: str, texts: list[str], query: bool = False, cache: bool = True) -> np.ndarray:
    """Unit-length embeddings; passage embeddings are cached on disk by content hash."""
    if query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    path = None
    if cache and not query:
        h = hashlib.sha1(("\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
        path = CACHE / f"emb_{name}_{h}.npy"
        if path.exists():
            return np.load(path)
    v = _encoder(name).encode(texts, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    if path is not None:
        np.save(path, v)
    return v


class Dense:
    def __init__(self, passages: list[tuple[str, str]], model: str = "bge-base"):
        self.name = model
        self.ids = [p[0] for p in passages]
        self.model = model
        self.matrix = embed(model, [p[1] for p in passages])

    def search(self, queries: list[str], k: int = 10) -> list[Ranking]:
        q = embed(self.model, queries, query=True, cache=False)
        sims = q @ self.matrix.T
        k = min(k, len(self.ids))
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out = []
        for row, idx in zip(sims, top):
            idx = idx[np.argsort(-row[idx])]
            out.append([(self.ids[i], float(row[i])) for i in idx])
        return out


# ------------------------------------------------------------------ hybrid (RRF)
class Hybrid:
    """Reciprocal-rank fusion: score = sum over retrievers of 1 / (60 + rank)."""

    def __init__(self, passages, dense_model: str = "bge-base", depth: int = 100, k0: int = 60):
        self.name = "hybrid"
        self.parts = [BM25(passages), Dense(passages, dense_model)]
        self.depth, self.k0 = depth, k0

    def search(self, queries: list[str], k: int = 10) -> list[Ranking]:
        runs = [p.search(queries, self.depth) for p in self.parts]
        out = []
        for per_q in zip(*runs):
            fused: dict[str, float] = {}
            for ranking in per_q:
                for r, (pid, _) in enumerate(ranking):
                    fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.k0 + r + 1)
            out.append(sorted(fused.items(), key=lambda x: -x[1])[:k])
        return out


# ------------------------------------------------------------------ reranker
@lru_cache(maxsize=None)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(RERANKER, device=_device(), max_length=512)
    if _device() == "cuda":
        m.model.half()
    return m


def rerank_scores(pairs: list[tuple[str, str]]) -> np.ndarray:
    if not pairs:
        return np.zeros(0)
    return np.asarray(_cross_encoder().predict(pairs, batch_size=32, show_progress_bar=False),
                      dtype=np.float32)


class Reranked:
    """First stage (hybrid by default) returns `pool` candidates; the cross-encoder reorders them."""

    def __init__(self, passages, first=None, pool: int = 30):
        self.name = "hybrid+rerank"
        self.text = dict(passages)
        self.first = first or Hybrid(passages)
        self.pool = pool

    def search(self, queries: list[str], k: int = 10) -> list[Ranking]:
        cands = self.first.search(queries, self.pool)
        pairs = [(q, self.text[pid]) for q, c in zip(queries, cands) for pid, _ in c]
        scores = rerank_scores(pairs)
        out, i = [], 0
        for c in cands:
            s = scores[i:i + len(c)]
            i += len(c)
            order = np.argsort(-s)[:k]
            out.append([(c[j][0], float(s[j])) for j in order])
        return out


def build(kind: str, passages: list[tuple[str, str]]):
    return {
        "bm25": lambda: BM25(passages),
        "bge-small": lambda: Dense(passages, "bge-small"),
        "bge-base": lambda: Dense(passages, "bge-base"),
        "hybrid": lambda: Hybrid(passages),
        "hybrid+rerank": lambda: Reranked(passages),
    }[kind]()
