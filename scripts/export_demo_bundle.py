from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noscope_bio.pipeline import DATA_DIR, export_demo_bundle


def main():
    artifacts = export_demo_bundle()
    print(f"Exported demo bundle to {DATA_DIR}")
    print("Evaluation metrics:")
    for key, value in artifacts["evaluation_metrics"].items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
