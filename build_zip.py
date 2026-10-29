"""Package the project for submission: code, results, figures, docs -- no venv, data, caches or weights.

    python build_zip.py   ->  dist/Verity_23215149_RuhinZaidi.zip
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".venv", "data", "models", "dist", "__pycache__", ".pytest_cache", ".git", "_private"}
SKIP_FILES = {".env"}

out = ROOT / "dist" / "Verity_23215149_RuhinZaidi.zip"
out.parent.mkdir(exist_ok=True)
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_dir() or set(rel.parts) & SKIP_DIRS or p.name in SKIP_FILES or p.suffix == ".pyc":
            continue
        z.write(p, Path("Verity") / rel)
        n += 1
print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
