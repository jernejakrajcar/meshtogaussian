"""Preverjanje CUDA/gsplat okolja pred zagonom treninga.

Skripta pove, ali so GPU, PyTorch, gsplat repozitorij in potrebne poti
pripravljeni, da se daljši trening ne ustavi šele na sredini.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.training.gsplat_runner import check_cuda_training_environment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether this machine is ready for the experimental CUDA/gsplat training path. "
            "The main project workflow uses Mesh2Splat-exported PLY LODs."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--gsplat-repo", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo = args.gsplat_repo or cfg.get("gsplat", {}).get("repo")
    status = check_cuda_training_environment(repo)
    print(json.dumps(status, indent=2))
    if status["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
