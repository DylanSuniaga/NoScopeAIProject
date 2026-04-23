"""Build the final report PDF.

Tries, in order:
  1. latexmk -pdf -bibtex -interaction=nonstopmode main.tex
  2. tectonic main.tex
  3. plain pdflatex + bibtex + pdflatex + pdflatex

Run from anywhere:

    python3 report/build.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
TEX_NAME = "main.tex"
PDF_NAME = "main.pdf"
FINAL_PDF = "group5_noscope_bio_report.pdf"


def _run(cmd: list[str]) -> int:
    print("  $", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPORT_DIR)


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def build() -> int:
    if not (REPORT_DIR / TEX_NAME).exists():
        print(f"ERROR: {REPORT_DIR / TEX_NAME} does not exist.", file=sys.stderr)
        return 1

    if _have("latexmk"):
        print("[build] using latexmk")
        rc = _run([
            "latexmk", "-pdf", "-bibtex",
            "-interaction=nonstopmode", "-halt-on-error",
            "-file-line-error", TEX_NAME,
        ])
    elif _have("tectonic"):
        print("[build] using tectonic")
        rc = _run(["tectonic", TEX_NAME])
    elif _have("pdflatex") and _have("bibtex"):
        print("[build] using pdflatex + bibtex manual loop")
        base = TEX_NAME[:-4]
        rc = _run(["pdflatex", "-interaction=nonstopmode", TEX_NAME])
        if rc == 0:
            rc = _run(["bibtex", base])
        if rc == 0:
            rc = _run(["pdflatex", "-interaction=nonstopmode", TEX_NAME])
        if rc == 0:
            rc = _run(["pdflatex", "-interaction=nonstopmode", TEX_NAME])
    else:
        print(
            "ERROR: no LaTeX toolchain found. Install one:\n"
            "  - tectonic: `brew install tectonic`\n"
            "  - TeX Live: `brew install --cask mactex` (or `mactex-no-gui`)\n",
            file=sys.stderr,
        )
        return 127

    if rc != 0:
        print(f"[build] LaTeX returned non-zero exit code {rc}", file=sys.stderr)
        return rc

    src = REPORT_DIR / PDF_NAME
    dst = REPORT_DIR / FINAL_PDF
    if src.exists():
        shutil.copyfile(src, dst)
        print(f"[build] wrote {dst}")
    else:
        print(f"[build] WARNING: expected {src} to exist after build", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(build())
