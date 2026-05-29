"""Skripta za izračun evalvacijskih metrik.

Primerja renderirane rezultate in izpiše številke, kot so MSE, PSNR in popping,
da lahko prehode in LOD nastavitve ocenimo bolj objektivno kot samo na oko.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.run_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline and print evaluation metrics.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    metrics = run_pipeline(args.config)
    metrics_path = Path(metrics["outputs"]["video"]).with_name("metrics.json")
    print(json.dumps({"metrics": str(metrics_path), "popping_score": metrics["transition"]["popping_score"]}, indent=2))


if __name__ == "__main__":
    main()
