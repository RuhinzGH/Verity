"""Publish Verity on the Hugging Face Hub (log in first: huggingface-cli login).

    python -m deploy.publish model    the fine-tuned verifier, with its model card
    python -m deploy.publish space    the app; the Space rebuilds itself from the Dockerfile

Only files tracked by git are uploaded to the Space, so nothing ignored (.env, _private/,
data, models) can leave the laptop this way.
"""
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from verity.config import FINETUNED_NLI, FINETUNED_NLI_HUB

ROOT = Path(__file__).resolve().parent.parent
SPACE = "RuhinZ/verity"
# Spaces read their settings from the front matter of README.md
SPACE_HEADER = """---
title: Verity
emoji: 🔎
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Check an AI-written answer claim by claim
---

"""


def publish_model(api: HfApi) -> None:
    if not (FINETUNED_NLI / "model.safetensors").exists():
        sys.exit(f"{FINETUNED_NLI} not found: train it first (python -m experiments.e03_finetune)")
    api.create_repo(FINETUNED_NLI_HUB, exist_ok=True)
    api.upload_folder(repo_id=FINETUNED_NLI_HUB, folder_path=str(FINETUNED_NLI),
                      commit_message="Fine-tuned verifier (experiment 3)")
    api.upload_file(repo_id=FINETUNED_NLI_HUB, path_or_fileobj=str(ROOT / "deploy" / "model_card.md"),
                    path_in_repo="README.md", commit_message="Model card")
    print(f"https://huggingface.co/{FINETUNED_NLI_HUB}")


def publish_space(api: HfApi) -> None:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    files = [f for f in tracked if f != "README.md" and not f.startswith(("colab/", ".github/"))]
    readme = SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    ops = [CommitOperationAdd(f, str(ROOT / f)) for f in files]
    ops.append(CommitOperationAdd("README.md", readme.encode("utf-8")))
    api.create_repo(SPACE, repo_type="space", space_sdk="docker", exist_ok=True)
    keep = set(files) | {"README.md", ".gitattributes"}
    ops += [CommitOperationDelete(f) for f in api.list_repo_files(SPACE, repo_type="space") if f not in keep]
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    api.create_commit(SPACE, ops, repo_type="space", commit_message=f"Deploy GitHub commit {head}")
    print(f"https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["model", "space"]
    api = HfApi()
    for w in what:
        {"model": publish_model, "space": publish_space}[w](api)
