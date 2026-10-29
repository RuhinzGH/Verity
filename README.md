# Verity: AI Answer Verification

**No strings. Only sources.**

Ruhin Zaidi · 23215149 · BCA (Honours) · GenAI mini project

[![tests](https://github.com/RuhinzGH/Verity/actions/workflows/tests.yml/badge.svg)](https://github.com/RuhinzGH/Verity/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

**Live app:** https://huggingface.co/spaces/RuhinZ/verity ·
**Fine-tuned verifier:** https://huggingface.co/RuhinZ/verity-deberta-scifact

Verity checks an AI-written answer the way a careful reader would. It breaks the answer
into separate claims, finds evidence for each one in a set of sources, decides whether
the evidence **supports** the claim, **refutes** it, or **doesn't settle it**, and cites
the passage it used. When the model cannot narrow a claim to one verdict at a
calibrated confidence level, Verity says **UNCERTAIN** instead of guessing.

Everything runs on a laptop: the verifier, the embeddings, the reranker and the local
language models fit on a 4 GB GPU, and the hosted LLM is optional. One extra experiment
(8) repeats the key comparisons on a cloud A100 GPU with much bigger models, to show what
the laptop version gives up.

---

## How it works

```
 AI answer ──► 1 DECOMPOSE ──► claims ──► 2 RETRIEVE ──► top passages ──► 3 WINDOW
               sentences, or an            BM25 + embeddings          2-sentence windows
               LLM that writes             (hybrid, reciprocal-       of each passage
               atomic claims               rank fusion)
                                                                            │
   report  ◄── 5 DECIDE ◄───────────────── 4 VERIFY ◄──────────────────────┘
   claims,     calibrated probabilities    DeBERTa-v3 NLI verifier,
   verdicts,   + conformal prediction      fine-tuned on SciFact on the
   citations   → verdict or UNCERTAIN      laptop GPU
```

| Step | Module | What it does |
|---|---|---|
| Decompose | `verity/decompose.py` | Sentence split, or an LLM rewrites sentences into atomic, self-contained claims and records which sentence each came from |
| Retrieve | `verity/retrieve.py` | BM25, bge-small / bge-base embeddings, hybrid (reciprocal-rank fusion), hybrid + cross-encoder reranker |
| Verify | `verity/verify.py` | NLI cross-encoders (zero-shot and fine-tuned) and LLM judges, all returning SUPPORTED / REFUTED / NEI probabilities |
| Decide | `verity/conformal.py` | Temperature scaling, then split conformal prediction: a label set that contains the true label 90% of the time on data like the calibration set; more than one label means UNCERTAIN |
| Pipeline | `verity/pipeline.py` | All of the above in one call: `Verity(sources).check(answer)` |
| Interface | `app.py` | Streamlit app: paste an answer and sources, see every claim coloured and cited |

## What was measured

Eight experiments (plus 3b) on three public benchmarks, each written to answer one question.
Full tables: **[RESULTS.md](RESULTS.md)**. Figures: `figures/`.

| # | Question | Data |
|---|---|---|
| 1 | Which retriever finds the evidence? | SciFact (5,183 abstracts), Climate-FEVER (5,240 sentences) |
| 2 | Which verifier labels a claim correctly when it has the right evidence? | SciFact dev, Climate-FEVER |
| 3 | Can a laptop GPU fine-tune a better verifier? | SciFact train |
| 3b | How much is luck? Does re-weighting the classes help? | SciFact train, 3 seeds |
| 4 | Can the uncertainty labels be trusted, and when do they break? | SciFact dev, Climate-FEVER |
| 5 | How much is lost when the system must find its own evidence? | SciFact dev, full corpus |
| 6 | Does it catch hallucinations in real LLM answers? | RAGTruth (answers from six LLMs, human-marked) |
| 7 | Where does it fail? | error analysis of experiment 2's predictions |
| 8 | Would a bigger model do better? A larger verifier, and LLM judges from 7B to 32B parameters | experiments 2 and 6 again, on a cloud A100 GPU |

<!-- results:start -->
**Headline results** (generated from `results/`; brackets are 95% bootstrap CIs)

| Measure | Result | Compared with |
|---|---|---|
| Retrieval, Climate-FEVER Recall@10 | BM25 0.458 | hybrid 0.575 |
| *Oracle rows: the verifier is handed the annotated evidence.* | | |
| Verifier macro-F1, SciFact dev (oracle evidence) | zero-shot 0.657 [0.598, 0.709] | fine-tuned 0.786 [0.737, 0.830] |
| Verifier macro-F1, Climate-FEVER, all 6,905 pairs (out of domain) | zero-shot 0.446 [0.426, 0.464] | fine-tuned 0.512 [0.490, 0.532] |
| Coverage of the 90% uncertainty labels, oracle evidence | SciFact dev 0.910 | Climate-FEVER 0.802 (shifted data) |
| *End-to-end rows: the system finds its own evidence in 5,183 abstracts.* | | |
| End-to-end macro-F1, SciFact dev (design chosen on held-out claims) | windows + verity-finetuned | 0.653 [0.595, 0.708] |
| Coverage of the 90% uncertainty labels, end to end | windows + verity-finetuned | 0.847 |
| RAGTruth hallucination-detection F1 (sentence claims + fine-tuned verifier, threshold fitted on RAGTruth train) | 0.686 [0.626, 0.740] | flag-every-answer baseline 0.667 |
<!-- results:end -->

## Running it

Requires Python 3.12. An NVIDIA GPU (4 GB is enough) is needed only to train the
verifier and for the LLM experiments; the app and the NLI experiments also run on a CPU.

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows; source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or /whl/cpu without a GPU
pip install -r requirements.txt

python -m verity.data                      # download the three benchmarks (parquet files)
python -m verity.models verifier           # download the fine-tuned verifier (~740 MB)
python -m verity.models                    # download the local LLMs (optional, ~9 GB)

streamlit run app.py                       # the interface, at http://localhost:8501
```

The verifier on the Hub is the one experiment 3 trained, unchanged; to train it yourself
instead, run `python -m experiments.e03_finetune` (needs the GPU). Without it the app falls
back to the zero-shot verifier.

The hosted LLM (Groq) is optional. To use it, copy `.env.example` to `.env` and add a key.
Without one, the experiments use the local Qwen models for every LLM step, and the app
checks sentence by sentence (its LLM claim-splitting option needs the key).

### Reproducing the experiments

```bash
python -m experiments.e01_retrieval
python -m experiments.e03_finetune         # ~15 min on an RTX 3050 laptop GPU
python -m experiments.e03b_seeds           # 3 seeds x 2 recipes, ~70 min (resumable)
python -m experiments.e02_verifiers        # add --no-llm to skip the LLM judges
python -m experiments.e04_conformal
python -m experiments.e05_end_to_end
python -m experiments.e05b_evidence_design
python -m experiments.e06_ragtruth
python -m experiments.e07_error_analysis
# experiment 8 needs a large GPU: open colab/verity_cloud_run.ipynb, then _part2 and
# _part3 on the same Colab A100 runtime. The saved notebooks are the record of that run
# (parts 1-2 show judge scores before the scoring fix, part 3 the re-scored ones);
# build_colab_notebook.py rebuilds them with EMPTY outputs, so build into a copy.
python -m experiments.make_figures
python -m experiments.write_results        # regenerates RESULTS.md and the table above
pytest                                     # offline: no network or GPU needed
```

Model outputs are cached in `data/cache/` (LLM replies, NLI scores, embeddings), so a
re-run is fast and every LLM answer used in the results can be inspected.

## Design decisions

- **Claims, not answers.** One wrong number in a long answer should not hide behind
  nine correct sentences, so verdicts are per claim and mapped back onto the sentence.
- **A small trained verifier, not only an LLM judge.** A 184M-parameter cross-encoder
  gives real probabilities, runs in milliseconds and can be calibrated; an LLM's stated
  confidence is only text. Both are compared in experiment 2.
- **Uncertainty with a guarantee, and a check of where it holds.** "UNCERTAIN" comes from
  conformal prediction, so the rate at which the true label is inside Verity's answer is
  controlled (90% on data like the calibration set) rather than a threshold picked by eye.
  The guarantee only covers data like the calibration set, so experiment 4 measures how
  far it breaks on a different kind of claim, and experiment 5b measures what it actually
  delivers on evidence the system retrieved itself.
- **No peeking at the test set.** The verifier was trained on SciFact train and
  calibrated on a held-out 20% of it. The evidence design for the full pipeline
  (experiment 5b) was chosen on that held-out part and only then run on dev; it was set
  up after a first run of experiment 5 on dev had shown a problem. RAGTruth thresholds
  come from its training split only.
- **An audit, and what it changed.** An independent review of the first version found
  that "not enough evidence" examples were always three sentences long while real
  rationales are one or two, so a model could partly spot them by length. The pairs were
  rebuilt with matched lengths, the verifier retrained and every experiment re-run; the
  first version's results are kept in `results/v1_before_audit/` for comparison. The fix
  changed a conclusion: in the first version the fine-tuned verifier's advantage
  vanished once the system had to find its own evidence. That turned out to be the
  shortcut at work (the pipeline fed it three-sentence chunks, which it had learned to
  call NEI). Without the shortcut, the fine-tuned verifier is ahead again end to end, by
  5-6 points with every retriever, though the confidence intervals overlap (experiments
  5, 5b).
- **One verifier, one evidence design.** Both modes use the fine-tuned verifier reading
  two-sentence windows. For the SciFact corpus, Verity first finds the three most relevant
  abstracts, then reads every window inside them: the design experiment 5b chose on
  held-out claims. Pasted sources are cut into windows the same way; that mode was not
  separately tuned.
- **A second bug, caught by checking a surprising result.** In experiment 8 the 14B LLM
  judge first scored far below the 7B. Instead of reporting it, its raw answers were
  checked: they were more accurate than the 7B's. The fault was in Verity's scoring: when
  an LLM stated a low confidence (for example "NEI, confidence 0.2"), the conversion to
  probabilities produced a three-way tie, and the tie was counted as SUPPORTED. The
  stated label is now always the one scored (with a regression test), and every LLM
  judge, on the laptop and in the cloud, was re-scored from its saved answers.
- **What a bigger model buys (experiment 8).** On the science claims it was trained for,
  the 184M-parameter fine-tuned verifier scores highest of everything that runs on the
  laptop and of every LLM judge tested up to 32B parameters: clearly above the 1.5B-7B
  judges, and about 3 points above the 14B and 32B judges, a gap within their
  uncertainty. A larger fine-tuned verifier (435M, cloud) adds a few points on science
  claims but nothing on climate claims. On climate claims, a kind of text the verifier
  was not trained on, LLM judges of 7B and above do better (the 3B is also higher, but
  within noise). Fine-tuning wins where it was trained; general models transfer better.
  Verity keeps the small verifier because it runs on a laptop and is the most accurate
  where its training applies.
- **Honest absence.** NEI ("not enough evidence") is a first-class answer. A claim the
  sources do not cover is not called false.

## Limitations

- The verifier checks claims against the *given* sources. If the sources are wrong,
  Verity will confidently agree with them.
- Fine-tuning used one science benchmark; on climate claims (a different style of
  text) accuracy and the coverage guarantee both drop (experiment 4).
- The fine-tuned verifier leans towards "not enough evidence": most of its mistakes on
  supported claims are NEI verdicts, many of them confident (experiment 7, exact counts
  in RESULTS.md). About 62% of its training pairs are NEI examples, the likely cause.
  Experiment 3b tried re-weighting the classes: it scored lower on the held-out claims, so
  it was not adopted, although on the test set it came out about 1.7 points higher with
  better recall on supported claims (3 seeds each). Choosing it after seeing that would
  have been choosing on the test set, so the standard recipe stays; class weighting is a
  reasonable next step to validate properly.
- On RAGTruth, checking an answer claim by claim catches most hallucinated answers but
  also flags most of the correct ones; its F1 cannot be told apart from flagging every
  answer (experiment 6). Precision is the weak side. The 3B LLM judge shows the opposite
  trade-off (more precise, but misses many hallucinations), and the 32B judge on the cloud
  GPU flags almost everything (it catches every hallucinated answer but also flags 81% of
  the correct ones). Spotting hallucinations in whole answers is not solved by any method
  tried here.
- The uncertainty labels' 90% target was calibrated on annotated evidence. On evidence
  the system retrieves itself, the measured coverage is lower (experiment 5b).
- In experiment 8, the 1.5B and 3B judges ran on the laptop (the 3B compressed to 4-bit)
  while the 7B, 14B and 32B ran uncompressed on the A100, so the size comparison also
  mixes in compression. LLM judges' stated confidence is not used consistently by the
  models, so their calibration (ECE) numbers are only indicative.
- The deployed verifier is one training run (seed 42). Experiment 3b repeats the recipe with
  three seeds: its test score moves by about 0.001 between seeds, so the result is not a
  lucky run. Other numbers come from seed 42 unless stated.
- Two SciFact dev claims also appear word for word in SciFact train (a property of the
  dataset); 2 of 300 test claims.
- The rule-based traits in the error analysis (numbers, negation, ...) are crude
  pattern matches, not linguistic annotation.
- Long sources are cut into overlapping two-sentence windows; claims that need facts
  from far-apart sentences can be missed.

## Data and models

Licences as listed on each card when the project was built; all are used here for
non-commercial coursework only.

| | Source | Licence |
|---|---|---|
| SciFact | Wadden et al., 2020 (`allenai/scifact`) | CC BY-NC 2.0 |
| Climate-FEVER | Diggelmann et al., 2020 (`tdiggelm/climate_fever`) | not stated on the dataset card |
| RAGTruth | Niu et al., 2024 (processed copy `wandb/RAGTruth-processed`) | not stated on this copy's card |
| Verifiers | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (base of the fine-tuned verifier), `cross-encoder/nli-deberta-v3-base` (zero-shot comparison) | MIT / Apache-2.0 |
| Embeddings, reranker | `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-reranker-base` | MIT |
| Local LLMs | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` | Apache-2.0 / Qwen Research licence |
| Cloud models (experiment 8) | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, `Qwen/Qwen2.5-7B-Instruct`, `-14B-Instruct`, `-32B-Instruct` | MIT / Apache-2.0 |

## Project layout

```
verity/          the package: data, retrieval, verification, conformal, pipeline
experiments/     one script per experiment, plus figures and RESULTS.md generation
colab/           the cloud-GPU notebooks for experiment 8, with their saved outputs
results/         JSON output of every experiment (v1_before_audit/: the superseded first run)
figures/         report figures
tests/           pytest suite (offline)
deploy/          the Hugging Face Space and model-card files, and the script that publishes them
app.py           Streamlit interface
Dockerfile       the image the live app runs in
.github/         GitHub Actions: the test suite on every push
```

## Deployment

The live app runs on a free Hugging Face Space (2 CPU cores, 16 GB memory) from the
`Dockerfile` in this repository. When the image is built it downloads the benchmarks and
the fine-tuned verifier, and computes the embeddings of the 5,183 SciFact abstracts once,
so a visitor never waits for the index to be built. The verifier runs on the CPU, which is
fast enough for one answer at a time.

```bash
huggingface-cli login                      # once, with a write token
python -m deploy.publish model             # upload models/verity-deberta-scifact to the Hub
python -m deploy.publish space             # upload the app; the Space rebuilds itself
```

The optional Groq key is stored as a Space secret (`GROQ_API_KEY`), never in the code. A
free Space sleeps after a period without visitors; the first visit afterwards takes about
a minute while it starts.

## Licence

The code is released under the [MIT licence](LICENSE). The datasets and models keep their
own licences (table above); the fine-tuned verifier is trained on SciFact (CC BY-NC 2.0)
and is therefore for non-commercial use.
