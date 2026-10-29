"""Build the self-contained notebooks for a large cloud GPU (experiment 8).

    python build_colab_notebook.py          ->  colab/verity_cloud_run.ipynb
    python build_colab_notebook.py part2    ->  colab/verity_cloud_run_part2.ipynb
    python build_colab_notebook.py part3    ->  colab/verity_cloud_run_part3.ipynb

WARNING: building overwrites the notebook with empty outputs. The notebooks in colab/
are the record of the cloud run (part 1 and 2 outputs show the judge scores before the
scoring fix; part 3 shows the re-scored ones), so build into a copy if you re-run.

Each notebook carries the project code inside it (a base64 zip of verity/ and
experiments/), downloads the datasets and models on the cloud machine, runs
experiment 8 and prints the results, so nothing has to be uploaded by hand.
Open it in VS Code, choose a Colab kernel with a large GPU, and Run All.
Parts 2 and 3 run on the same server after part 1 and merge into its results.
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "colab" / "verity_cloud_run.ipynb"
OUT2 = ROOT / "colab" / "verity_cloud_run_part2.ipynb"
NL = "\n"

PIP = ("!pip -q install 'transformers==4.46.3' 'sentence-transformers==3.3.1' 'tokenizers<0.21' "
       "'accelerate==1.1.1' 'datasets==3.1.0' 'huggingface-hub<1.0' bm25s==0.3.15 PyStemmer "
       "scikit-learn scipy pandas python-dotenv")
PRINT_RESULTS = NL.join(["# the results, printed so they are saved in this notebook",
                         "import json",
                         "r = json.load(open('/content/verity/results/e08_scale_up.json'))",
                         "print('RESULTS-JSON-START')", "print(json.dumps(r))", "print('RESULTS-JSON-END')"])


def code_zip() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("verity", "experiments"):
            for p in sorted((ROOT / folder).rglob("*.py")):
                z.write(p, p.relative_to(ROOT))
    return base64.b64encode(buf.getvalue()).decode()


def unpack_cell() -> str:
    return NL.join(["# unpack the project code carried inside this notebook",
                    "import base64, io, zipfile, os",
                    f"CODE = '{code_zip()}'",
                    "os.makedirs('/content/verity', exist_ok=True)",
                    "zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE))).extractall('/content/verity')",
                    "os.chdir('/content/verity')",
                    "print(sorted(os.listdir('.')))"])


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def write(path, cells):
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", path, f"({path.stat().st_size / 1e3:.0f} kB)")


def main():
    write(OUT, [
        cell("markdown", "# Verity: scale-up run (experiment 8)" + NL + NL +
             "Runs the same experiments as on the 4 GB laptop GPU with larger models: a DeBERTa-v3-large "
             "verifier (3 seeds) and Qwen2.5 7B / 14B LLM judges. Choose a runtime with a large GPU "
             "(A100 preferred, L4 works), then Run All. The last cell prints the results."),
        cell("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        cell("code", unpack_cell()),
        cell("code", PIP),
        cell("code", "%cd /content/verity" + NL + "!python -m verity.data"),
        cell("code", "# A: DeBERTa-v3-large verifier, 3 seeds (about 20-40 min on an A100)" + NL +
             "!python -m experiments.e08_scale_up --only=large 2>&1 | grep -v -i warn"),
        cell("code", "# B: Qwen2.5 7B and 14B as LLM judges (about 15-30 min on an A100)" + NL +
             "!python -m experiments.e08_scale_up --only=judges 2>&1 | grep -v -i warn"),
        cell("code", PRINT_RESULTS),
    ])


def part2():
    write(OUT2, [
        cell("markdown", "# Verity: cloud run, part 2 (Qwen2.5-32B)" + NL + NL +
             "Run on the same A100 server after part 1 has finished (about 30-45 minutes). "
             "Data and part-1 results on the server are kept; the last cell prints all experiment-8 results."),
        cell("code", "!nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv"),
        cell("code", unpack_cell()),
        cell("code", PIP + NL + "%cd /content/verity" + NL + "!python -m verity.data"),
        cell("code", "# Check: is the 14B judge's low score a bug? (raw answers + 50 claims unbatched, ~5 min)" + NL +
             "!python -m experiments.e08_scale_up --only=check14 2>&1 | grep -v -i warn | tail -20"),
        cell("code", "# C1: Qwen2.5-32B as a claim judge (SciFact dev + Climate-FEVER sample)" + NL +
             "!python -m experiments.e08_scale_up --only=judge32 2>&1 | grep -v -i warn | tail -20"),
        cell("code", "# C2: Qwen2.5-32B judging whole RAGTruth answers against their sources" + NL +
             "!python -m experiments.e08_scale_up --only=ragtruth32 2>&1 | grep -v -i warn | tail -20"),
        cell("code", PRINT_RESULTS),
    ])


def part3():
    """Re-score the LLM judges from the answers cached on the server (about a minute, no
    generation) after the scoring fix in verity.verify.judgement_probs."""
    write(ROOT / "colab" / "verity_cloud_run_part3.ipynb", [
        cell("markdown", "# Verity: cloud run, part 3 (re-score, ~1 minute)" + NL + NL +
             "Run on the same server as parts 1 and 2. No new generation: it re-scores the LLM judges' "
             "cached answers with the corrected scoring."),
        cell("code", unpack_cell()),
        cell("code", "%cd /content/verity" + NL + "!ls data/cache | grep llm_" + NL +
             "!python -m experiments.e08_scale_up --only=rescore 2>&1 | grep -v -i warn | tail -12"),
        cell("code", PRINT_RESULTS),
    ])


if __name__ == "__main__":
    {"part2": part2, "part3": part3}.get(next((a for a in sys.argv[1:]), ""), main)()
