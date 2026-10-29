"""Shared fixtures: keep the suite offline, CPU-only and away from the real caches."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" is dropped on Windows
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to open a non-local connection fails loudly."""
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "localhost", "::1"):
            return real_connect(self, address)
        raise RuntimeError(f"network access attempted in tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(autouse=True)
def llm_cache_dir(monkeypatch, tmp_path):
    """Point the LLM response cache at a temp dir so data/cache/llm_*.jsonl is never read or written."""
    from verity import llm
    d = tmp_path / "llm_cache"
    d.mkdir()
    real_cache = llm._cache                 # a test may monkeypatch llm._cache itself
    monkeypatch.setattr(llm, "CACHE", d)
    real_cache.cache_clear()
    yield d
    real_cache.cache_clear()
