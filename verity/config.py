"""Paths, model names and the label set shared by every part of Verity."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"            # embeddings, LLM responses, NLI scores
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"          # the fine-tuned verifier is written here

for _p in (CACHE, RESULTS, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# One label set for every dataset. NEI = "not enough evidence".
LABELS = ["SUPPORTED", "REFUTED", "NEI"]
L2I = {l: i for i, l in enumerate(LABELS)}

# Retrieval
BM25_STEMMER = "english"
DENSE_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
}
# bge models are trained with an instruction on the query side only
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANKER = "BAAI/bge-reranker-base"

# Verification (natural language inference). Each maps its own label names onto ours.
NLI_MODELS = {
    # trained on MNLI + SNLI only: the "general" verifier
    "deberta-mnli": "cross-encoder/nli-deberta-v3-base",
    # also trained on FEVER-NLI and ANLI: a fact-checking verifier
    "deberta-fever": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
}
FINETUNED_NLI = MODELS / "verity-deberta-scifact"
FINETUNED_NLI_HUB = "RuhinZ/verity-deberta-scifact"   # the same weights, for the live app and fresh clones

# LLMs (Groq API; 1,000 requests/day and 8,000 tokens/min per model on this account)
LLM_DECOMPOSER = "openai/gpt-oss-20b"
LLM_JUDGE = "openai/gpt-oss-120b"
# Local models (laptop GPU, 4 GB): downloaded into models/hf/ by `python -m verity.models`
LOCAL_LLM = "local:Qwen2.5-1.5B-Instruct"          # fp16, ~3.1 GB
LOCAL_LLM_LARGE = "local:Qwen2.5-3B-Instruct"      # 4-bit, ~2.2 GB

SEED = 42
