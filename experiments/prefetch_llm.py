"""Fill the LLM response cache ahead of experiments 2 and 6 (API calls only, no GPU).

    python -m experiments.prefetch_llm judge        # gpt-oss-120b judgements for e02 + e06
    python -m experiments.prefetch_llm decompose    # gpt-oss-20b decompositions for e06

The Groq limit is per model, so the two can run at the same time. Responses land in
data/cache/llm_<model>.jsonl; the experiments then read them instead of calling the API.
"""
import sys

from verity import decompose as D, pairs as P
from verity.config import LLM_DECOMPOSER, LLM_JUDGE
from verity.verify import LLMJudge

from experiments.e02_verifiers import stratified
from experiments.e06_ragtruth import llm_judge, sample


def judge():
    j = LLMJudge(LLM_JUDGE)
    for name, pairs in [("scifact_dev", P.scifact_pairs("dev")),
                        ("climate_fever_sample", stratified(P.climate_pairs(), 300))]:
        for i, p in enumerate(pairs):
            j.judge(p["evidence"], p["claim"])
            if i % 50 == 0:
                print(name, i, flush=True)
    llm_judge(sample("test"))
    print("judge done", flush=True)


def decompose():
    for split in ["train", "test"]:
        for i, r in enumerate(sample(split)):
            D.by_llm(r.output, LLM_DECOMPOSER)
            if i % 50 == 0:
                print(split, i, flush=True)
    print("decompose done", flush=True)


if __name__ == "__main__":
    {"judge": judge, "decompose": decompose}[sys.argv[1]]()
