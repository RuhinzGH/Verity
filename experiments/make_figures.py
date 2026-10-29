"""Figures for the report, drawn from results/*.json only (no number is typed in by hand).

    python -m experiments.make_figures

Writes figures/fig*.png at 200 dpi. Colours follow one fixed categorical order, so a
method keeps its colour in every figure.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from verity.config import FIGURES, RESULTS  # noqa: E402

# validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# one colour per verifier, used in every figure
VERIFIER_COLOUR = {"deberta-mnli": C[0], "deberta-fever": C[1], "verity-finetuned": C[2],
                   "qwen2.5-1.5b": C[3], "qwen2.5-3b": C[4], "qwen2.5-32b": C[6], "gpt-oss-120b": C[7]}


def method_colour(name):
    """RAGTruth methods are '<decomposer> + <verifier>' or '<llm> judge ...': colour by the
    part that does the judging."""
    for v, c in VERIFIER_COLOUR.items():
        if name.endswith(v) or name.startswith(v + " judge"):
            return c
    return MUTED

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 10.5, "axes.titlecolor": INK,
})


def load(name):
    p = RESULTS / f"{name}.json"
    if not p.exists() and name == "e06_ragtruth":          # older partial-run file name
        p = RESULTS / "e06_ragtruth_sentences.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.savefig(FIGURES / name, facecolor="white")
    plt.close(fig)
    print("wrote", FIGURES / name)


def bar_labels(ax, bars, fmt="{:.2f}", above=None):
    """Value above each bar; `above` lifts a label over its error bar."""
    for i, b in enumerate(bars):
        top = b.get_height() + (above[i] if above else 0)
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, top),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)


# ------------------------------------------------------------------ 1 retrieval
def fig_retrieval():
    r = load("e01_retrieval")
    if not r:
        return
    methods = list(r["scifact_dev"]["methods"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (key, title) in zip(axes, [("scifact_dev", "SciFact (5,183 abstracts)"),
                                       ("climate_fever", "Climate-FEVER (5,240 sentences)")]):
        ms = ["recall@1", "recall@5", "recall@10", "ndcg@10"]
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            vals = [r[key]["methods"][m][k] for k in ms]
            xs = [j + (i - (len(methods) - 1) / 2) * w for j in range(len(ms))]
            ax.bar(xs, vals, w * 0.92, color=C[i], label=m)
        ax.set_xticks(range(len(ms)), ["Recall@1", "Recall@5", "Recall@10", "nDCG@10"])
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("score (higher is better)")
    axes[1].legend(loc="upper left", fontsize=8, ncol=1)
    fig.suptitle("Experiment 1: finding the evidence", fontweight="bold", x=0.08, ha="left", y=1.04)
    save(fig, "fig1_retrieval.png")


# ------------------------------------------------------------------ 2 verifiers (oracle)
def fig_verifiers():
    r = load("e02_verifiers")
    if not r:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (key, title) in zip(axes, [("scifact_dev", "SciFact dev (in domain), n = 300"),
                                       ("climate_fever_sample", "Climate-FEVER (out of domain), n = 300")]):
        names = [n for n in r[key] if not n.startswith("_")]
        vals = [r[key][n]["macro_f1"] for n in names]
        lo = [v - r[key][n]["macro_f1_ci95"][0] for n, v in zip(names, vals)]
        hi = [r[key][n]["macro_f1_ci95"][1] - v for n, v in zip(names, vals)]

        cols = [VERIFIER_COLOUR.get(n, MUTED) for n in names]
        bars = ax.bar(range(len(names)), vals, 0.62, color=cols, yerr=[lo, hi],
                      error_kw={"ecolor": INK, "elinewidth": 0.9, "capsize": 3})
        bar_labels(ax, bars, above=hi)
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
        ax.axhline(1 / 3, color=MUTED, lw=0.9, ls="--")
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("macro-F1 (95% bootstrap CI)")
    axes[0].annotate("chance (1/3)", (-0.45, 1 / 3 + 0.012), fontsize=7.5, color=MUTED, ha="left")
    fig.suptitle("Experiment 2: labelling a claim, given the right evidence", fontweight="bold", x=0.08, ha="left", y=1.04)
    save(fig, "fig2_verifiers.png")


# ------------------------------------------------------------------ 3 calibration
def fig_calibration():
    r = load("e04_conformal")
    if not r:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls="--", label="perfect")
    for i, m in enumerate(["deberta-fever", "verity-finetuned"]):
        t = r["models"][m]["tests"]["scifact_dev"]
        b = t["reliability_after"]
        ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], "-o", ms=4, lw=2,
                color=C[1 + i], label=f"{m} (ECE {t['ece_after']:.3f})")
    ax.set(xlabel="stated confidence", ylabel="actual accuracy", title="Reliability, SciFact dev", xlim=(0.3, 1), ylim=(0, 1))
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    for i, m in enumerate(["deberta-fever", "verity-finetuned"]):
        rc = r["models"][m]["tests"]["scifact_dev"]["risk_coverage"]
        ax.plot([x["coverage"] for x in rc], [x["accuracy"] for x in rc], lw=2, color=C[1 + i], label=m)
    ax.set(xlabel="share of claims answered (most confident first)", ylabel="accuracy on those",
           title="Answer less, be right more", ylim=(0.4, 1.02))
    ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[2]
    m = r["models"]["verity-finetuned"]["tests"]
    alphas = ["0.05", "0.1", "0.2"]
    target = [1 - float(a) for a in alphas]
    x = range(len(alphas))
    ax.plot(x, target, "--", color=MUTED, lw=0.9, label="promised coverage")
    for i, (k, lab) in enumerate([("scifact_dev", "SciFact dev (same distribution)"),
                                  ("climate_fever", "Climate-FEVER (shifted)")]):
        ax.plot(x, [m[k]["conformal"][a]["coverage"] for a in alphas], "-o", ms=5, lw=2, color=C[1 + i * 1 + 1], label=lab)
    ax.set_xticks(list(x), [f"{1 - float(a):.0%}" for a in alphas])
    ax.set(xlabel="target coverage (1 - alpha)", ylabel="actual coverage", title="Does the coverage promise hold?", ylim=(0.4, 1))
    ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Experiment 4: can the uncertainty labels be trusted?", fontweight="bold", x=0.06, ha="left")
    fig.tight_layout()
    save(fig, "fig3_calibration.png")


# ------------------------------------------------------------------ 4 end to end
def fig_end_to_end():
    o, e = load("e02_verifiers"), load("e05_end_to_end")
    b = load("e05b_evidence_design")
    if not (o and e):
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    rows = [("oracle evidence", {v: o["scifact_dev"][v]["macro_f1"] for v in ["deberta-fever", "verity-finetuned"]})]
    for ret in ["bm25", "bge-base", "hybrid"]:
        rows.append((f"{ret}, top3-concat", {v: e["runs"][f"{ret} + {v}"]["macro_f1"] for v in ["deberta-fever", "verity-finetuned"]}))
    if b:
        for d in ["per-sentence", "windows"]:
            rows.append((f"hybrid, {d}", {v: b["dev"][f"{d} + {v}"]["macro_f1"] for v in ["deberta-fever", "verity-finetuned"]}))
    w = 0.38
    for i, v in enumerate(["deberta-fever", "verity-finetuned"]):
        bars = ax.bar([j + (i - 0.5) * w for j in range(len(rows))], [r[1][v] for r in rows], w * 0.92,
                      color=C[1 + i], label=v)
        bar_labels(ax, bars)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=20, ha="right")
    ax.set(ylabel="macro-F1, SciFact dev", ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Experiment 5: the full pipeline vs. being handed the evidence", loc="left")
    save(fig, "fig4_end_to_end.png")


# ------------------------------------------------------------------ 5 RAGTruth
def fig_ragtruth():
    r = load("e06_ragtruth")
    if not r:
        return
    methods = dict(r["methods"])
    r8 = load("e08_scale_up")
    if r8 and "ragtruth_judge_32b" in r8:            # same 240 answers, run on the cloud GPU
        methods["qwen2.5-32b judge (cloud, no decomposition)"] = r8["ragtruth_judge_32b"]
    names = list(methods)
    tasks = ["QA", "Summary", "Data2txt"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [methods[n]["response"]["f1"]] + [methods[n]["by_task"][t]["f1"] for t in tasks]
        ax.bar([j + (i - (len(names) - 1) / 2) * w for j in range(4)], vals, w * 0.92, color=method_colour(n), hatch=("///" if n.startswith("qwen2.5-1.5b +") else None), edgecolor="white", label=n)
    ax.axhline(r["always_flag_baseline"]["f1"], color=MUTED, lw=0.9, ls="--")
    ax.annotate("flag everything", (3.45, r["always_flag_baseline"]["f1"] + 0.01), fontsize=7.5, color=MUTED, ha="right")
    ax.set_xticks(range(4), ["all answers"] + tasks)
    ax.set(ylabel="hallucination-detection F1", ylim=(0, 1))
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"Experiment 6: catching hallucinations in real LLM answers (RAGTruth, n = {r['n_test']})", loc="left")
    save(fig, "fig5_ragtruth.png")


# ------------------------------------------------------------------ 6 laptop vs cloud
def fig_scale():
    r8, r2 = load("e08_scale_up"), load("e02_verifiers")
    if not (r8 and r2):
        return
    rows = []          # (label, params in billions, scifact, climate, where, kind)
    for name, size in [("qwen2.5-1.5b", 1.5), ("qwen2.5-3b", 3.0)]:
        if name in r2["scifact_dev"]:
            c = r2["climate_fever_sample"].get(name, {}).get("macro_f1")
            rows.append((name, size, r2["scifact_dev"][name]["macro_f1"], c, "laptop", "llm"))
    for name, j in r8.get("judges", {}).items():
        if name == "qwen2.5-14b" and "check_14b" not in r8:
            continue                     # suspected bug, not plotted until checked
        size = float(name.split("-")[-1].rstrip("b"))
        rows.append((name, size, j["scifact_dev"]["macro_f1"], j["climate_fever_sample"]["macro_f1"], "cloud", "llm"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (idx, title, key) in zip(axes, [(2, "SciFact dev (in domain for the fine-tuned models)", "scifact_dev"),
                                            (3, "Climate-FEVER sample (out of domain)", "climate_fever_sample")]):
        llm = sorted([r for r in rows if r[idx] is not None], key=lambda r: r[1])
        ax.plot([r[1] for r in llm], [r[idx] for r in llm], "-o", color=C[4], lw=2, ms=6, label="Qwen2.5 LLM judge")
        for r in llm:
            ax.annotate(r[0].replace("qwen2.5-", ""), (r[1], r[idx]), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.5, color=MUTED)
        ax.axhline(r2[key]["verity-finetuned"]["macro_f1"], color=C[2], lw=2, label="fine-tuned base, 0.18B (laptop)")
        if "large_summary" in r8:
            ax.axhline(r8["large_summary"][key]["mean"], color=C[2], lw=2, ls="--", label="fine-tuned large, 0.44B (cloud)")
        ax.set_xscale("log")
        ax.set_xticks([1.5, 3, 7, 14, 32], ["1.5B", "3B", "7B", "14B", "32B"])
        ax.minorticks_off()
        ax.set(title=title, xlabel="LLM judge size (parameters, log scale)", ylim=(0, 1))
    axes[0].set_ylabel("macro-F1 (oracle evidence)")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Experiment 8: does a bigger model beat the small fine-tuned verifier?", fontweight="bold",
                 x=0.06, ha="left", y=1.03)
    save(fig, "fig6_scale.png")


if __name__ == "__main__":
    for f in [fig_retrieval, fig_verifiers, fig_calibration, fig_end_to_end, fig_ragtruth, fig_scale]:
        f()
