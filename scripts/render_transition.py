from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.run_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the transition video.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    metrics = run_pipeline(args.config)
    print(metrics["outputs"]["video"])


if __name__ == "__main__":
    main()
