---
license: cc-by-nc-2.0
language: en
base_model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
datasets:
- allenai/scifact
pipeline_tag: text-classification
tags:
- fact-checking
- natural-language-inference
- claim-verification
---

# verity-deberta-scifact

The claim verifier of [Verity](https://github.com/RuhinzGH/Verity), which checks an
AI-written answer claim by claim against sources. Given an evidence passage and a claim,
it says whether the evidence **supports** the claim (`entailment`), **refutes** it
(`contradiction`), or does not settle it (`neutral`, shown in Verity as "not enough evidence").

## Training

Fine-tuned from `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (184M parameters) on
SciFact train (Wadden et al., 2020), on a laptop GPU (RTX 3050, 4 GB), in about 9 minutes.

| | |
|---|---|
| Training pairs | 1,051 (80% of SciFact train claims; 263 supported, 140 refuted, 648 not enough evidence) |
| Calibration pairs | 263 (the other 20%, split by claim; never trained on) |
| Epochs, learning rate, batch | 3, 2e-5, 16 |
| Max length | 384 tokens |
| Seed | 42 |

## Results

On SciFact dev (300 claims, never seen in training), with the annotated evidence:
**macro-F1 0.786** (95% bootstrap interval 0.737 to 0.830), against 0.657 for the base
model. Three training seeds give 0.786 ± 0.001. Out of domain (Climate-FEVER) it drops to
0.512. Full results and method: the Verity repository's RESULTS.md.

## Use

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("RuhinZ/verity-deberta-scifact")
model = AutoModelForSequenceClassification.from_pretrained("RuhinZ/verity-deberta-scifact")
enc = tok("Mars is the fourth planet from the Sun.", "Mars is the third planet.", return_tensors="pt")
print(model.config.id2label[int(model(**enc).logits.argmax())])   # contradiction
```

Evidence goes first, the claim second.

## Limitations

- It leans towards "not enough evidence": about 62% of its training pairs are of that
  class, and most of its mistakes on supported claims are that verdict.
- Trained on biomedical abstracts only; accuracy is much lower on other kinds of text.
- It judges only against the evidence it is given. If the evidence is wrong, so is it.

## Licence

Trained on SciFact (CC BY-NC 2.0), so this model is for non-commercial use.
