"""Publish the verifier and the SciFact index on the Hugging Face Hub, where the live app
and fresh clones download them from (log in first: huggingface-cli login).

    python -m deploy.publish model    the fine-tuned verifier, with its model card
    python -m deploy.publish index    the SciFact corpus embeddings (computed here once)
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

from verity.config import CACHE, FINETUNED_NLI, FINETUNED_NLI_HUB, INDEX_HUB
from verity.models import scifact_index_name

ROOT = Path(__file__).resolve().parent.parent


def publish_model(api: HfApi) -> None:
    if not (FINETUNED_NLI / "model.safetensors").exists():
        sys.exit(f"{FINETUNED_NLI} not found: train it first (python -m experiments.e03_finetune)")
    api.create_repo(FINETUNED_NLI_HUB, exist_ok=True)
    api.upload_folder(repo_id=FINETUNED_NLI_HUB, folder_path=str(FINETUNED_NLI),
                      commit_message="Fine-tuned verifier (experiment 3)")
    api.upload_file(repo_id=FINETUNED_NLI_HUB, path_or_fileobj=str(ROOT / "deploy" / "model_card.md"),
                    path_in_repo="README.md", commit_message="Model card")
    print(f"https://huggingface.co/{FINETUNED_NLI_HUB}")


def publish_index(api: HfApi) -> None:
    name = scifact_index_name()
    if not (CACHE / name).exists():
        sys.exit(f"{CACHE / name} not found: run the app once in SciFact mode to compute it")
    api.create_repo(INDEX_HUB, repo_type="dataset", exist_ok=True)
    api.upload_file(repo_id=INDEX_HUB, repo_type="dataset", path_or_fileobj=str(CACHE / name),
                    path_in_repo=name, commit_message="SciFact corpus embeddings (bge-base)")
    print(f"https://huggingface.co/datasets/{INDEX_HUB}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["model", "index"]
    api = HfApi()
    for w in what:
        {"model": publish_model, "index": publish_index}[w](api)
