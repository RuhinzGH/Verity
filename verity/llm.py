"""LLM access: Groq's hosted models, or a small local model on the GPU.

Every response is cached on disk (data/cache/llm_<model>.jsonl, keyed by model + prompt), so
experiments re-run for free and results can be audited. The API key is read from the
GROQ_API_KEY environment variable or a .env file; it is never written anywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from functools import lru_cache

from .config import CACHE, MODELS



def _cache_file(model: str):
    """One file per model, so runs against different models can write at the same time."""
    return CACHE / ("llm_" + model.replace("/", "_").replace(":", "_") + ".jsonl")
_lock = threading.Lock()


@lru_cache(maxsize=None)
def _cache() -> dict:
    out = {}
    for f in sorted(CACHE.glob("llm_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                out[r["key"]] = r["response"]
            except (ValueError, KeyError):
                pass
    return out


def _key(model: str, prompt: str, max_tokens: int) -> str:
    return hashlib.sha256(json.dumps([model, prompt, max_tokens]).encode("utf-8")).hexdigest()


def _store(key: str, model: str, response: str) -> None:
    with _lock:
        _cache()[key] = response
        with _cache_file(model).open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": model, "response": response}) + "\n")


def api_available() -> bool:
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.environ.get("GROQ_API_KEY"))


class LLM:
    def __init__(self, model: str):
        self.model = model
        self.local = model.startswith("local:")
        self._client = None

    # -------------------------------------------------------------- public
    def complete(self, prompt: str, max_tokens: int = 800) -> str:
        key = _key(self.model, prompt, max_tokens)
        hit = _cache().get(key)
        if hit is not None:
            return hit
        text = self._local(prompt, max_tokens) if self.local else self._groq(prompt, max_tokens)
        _store(key, self.model, text)
        return text

    def complete_many(self, prompts: list[str], max_tokens: int = 800, batch_size: int = 8,
                      progress: str = "") -> list[str]:
        """Same as complete() for a list. Local models generate in batches (several times
        faster on a GPU); answers are cached one by one, so an interrupted run resumes."""
        keys = [_key(self.model, p, max_tokens) for p in prompts]
        todo = [i for i, k in enumerate(keys) if _cache().get(k) is None]
        if not self.local:
            for i in todo:
                self.complete(prompts[i], max_tokens)
        else:
            # similar lengths together: less padding per batch
            todo.sort(key=lambda i: len(prompts[i]))
            for b in range(0, len(todo), batch_size):
                idx = todo[b:b + batch_size]
                for i, text in zip(idx, self._local_batch([prompts[i] for i in idx], max_tokens)):
                    _store(keys[i], self.model, text)
                if progress:
                    print(f"  {progress}: {min(b + batch_size, len(todo))}/{len(todo)} generated", flush=True)
        return [_cache()[k] for k in keys]

    # -------------------------------------------------------------- Groq
    def _groq(self, prompt: str, max_tokens: int) -> str:
        from groq import Groq, RateLimitError, APIStatusError
        if self._client is None:
            from dotenv import load_dotenv
            load_dotenv()
            if not os.environ.get("GROQ_API_KEY"):
                raise RuntimeError("Set GROQ_API_KEY (see .env.example) or use the local model.")
            self._client = Groq(max_retries=0)
        kw = {"reasoning_effort": "low"} if "gpt-oss" in self.model else {}
        for attempt in range(12):
            try:
                r = self._client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=max_tokens, **kw)
                return r.choices[0].message.content or ""
            except RateLimitError as e:
                wait = float(e.response.headers.get("retry-after", 0) or 0) or min(60, 4 * 2 ** attempt)
                if "per day" in str(e).lower() or "tpd" in str(e).lower() or "rpd" in str(e).lower():
                    raise RuntimeError(f"Daily Groq quota reached for {self.model}: {str(e)[:200]}")
                time.sleep(wait + 0.5)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < 5:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("Groq kept rate-limiting; try again later.")

    # -------------------------------------------------------------- local (transformers, GPU)
    def _local(self, prompt: str, max_tokens: int) -> str:
        tok, model = _local_model(self.model.removeprefix("local:"))
        import torch
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def _local_batch(self, prompts: list[str], max_tokens: int) -> list[str]:
        tok, model = _local_model(self.model.removeprefix("local:"))
        import torch
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                         tokenize=False) for p in prompts]
        tok.padding_side = "left"                       # decoder-only: pad on the left
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        n = enc["input_ids"].shape[1]
        return [tok.decode(o[n:], skip_special_tokens=True) for o in out]


@lru_cache(maxsize=1)                    # one local model in GPU memory at a time
def _local_model(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # downloaded copy in models/hf/ if there is one, else a Hugging Face id (e.g. on a cloud GPU)
    path = MODELS / "hf" / name.split("/")[-1]
    path = path if path.exists() else name
    tok = AutoTokenizer.from_pretrained(path)
    if not torch.cuda.is_available():
        return tok, AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    if torch.cuda.get_device_properties(0).total_memory > 20e9:   # big GPU: full bf16, no quantisation
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
        return tok, model.eval()
    if "3B" in name or "7B" in name:     # 4-bit so it fits a 4 GB laptop GPU
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=q, device_map="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to("cuda")
    return tok, model.eval()
