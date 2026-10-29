"""Download models into models/ (plain files, no symlinks -- Windows-safe).

    python -m verity.models verifier    the fine-tuned verifier (~740 MB, from the Hugging Face Hub)
    python -m verity.models index       the SciFact corpus embeddings (16 MB; about an hour to compute on a CPU)
    python -m verity.models             the local LLMs (optional, ~9 GB)

The NLI, embedding and reranker models download themselves on first use.
"""
import hashlib
import sys

from huggingface_hub import snapshot_download

from .config import CACHE, FINETUNED_NLI, FINETUNED_NLI_HUB, INDEX_HUB, MODELS

LOCAL = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]


def fetch_verifier() -> str:
    """The verifier experiment 3 trained, published unchanged on the Hub."""
    return snapshot_download(FINETUNED_NLI_HUB, local_dir=str(FINETUNED_NLI),
                             allow_patterns=["*.json", "*.safetensors", "*.model"])


def fetch_index() -> str:
    """Passage embeddings are cached by content hash (retrieve.embed), so the published
    file is picked up as if it had been computed here."""
    return snapshot_download(INDEX_HUB, repo_type="dataset", local_dir=str(CACHE),
                             allow_patterns=["emb_*.npy"])


def scifact_index_name() -> str:
    """The file retrieve.embed caches for Verity.scifact(): the same texts, in the same
    order, hashed the same way."""
    from .data import scifact_corpus
    texts = [f"{d.title}. {d.text}" for d in scifact_corpus().values()]
    return f"emb_bge-base_{hashlib.sha1(chr(0).join(texts).encode('utf-8')).hexdigest()[:16]}.npy"


def ensure_app_assets() -> None:
    """Download whatever the app needs and is missing (benchmarks, verifier, SciFact index)."""
    from .data import RAW, download
    if not any(RAW.glob("scifact__corpus_*.parquet")):
        download()
    if not (FINETUNED_NLI / "model.safetensors").exists():
        fetch_verifier()
    if not (CACHE / scifact_index_name()).exists():
        fetch_index()


if __name__ == "__main__":
    if sys.argv[1:] == ["verifier"]:
        print(fetch_verifier())
    elif sys.argv[1:] == ["index"]:
        print(fetch_index())
    else:
        for m in LOCAL:
            print(snapshot_download(m, local_dir=str(MODELS / "hf" / m.split("/")[1]),
                                    allow_patterns=["*.json", "*.safetensors", "*.txt"]))
