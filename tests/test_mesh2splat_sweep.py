"""Testi za eksperimentalni sweep parametrov Mesh2Splat

Preverja se, da skripta zna v zagonu zapisati manifest in da metrika
pokritosti splattov vrača uporabne številke za izbiro gostote.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from src.gaussian.coverage import analyze_gaussian_coverage
from src.gaussian.model import GaussianCloud


def test_gaussian_coverage_reports_nn_to_scale() -> None:
    cloud = GaussianCloud(
        xyz=torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32),
        scale=torch.tensor([[0.05, 0.05, 0.001], [0.05, 0.05, 0.001], [0.05, 0.05, 0.001]], dtype=torch.float32),
        color=torch.ones((3, 3), dtype=torch.float32),
        opacity=torch.ones((3, 1), dtype=torch.float32),
    )
    stats = analyze_gaussian_coverage(cloud, sample_count=3)
    assert stats.count == 3
    assert stats.nn_to_scale_percentiles["p50"] > 1.0


def test_mesh2splat_sweep_dry_run_writes_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "demo_mesh2splat_sweep.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_mesh2splat.py",
            "--mesh",
            str(tmp_path / "demo.glb"),
            "--densities",
            "0.5,1.0",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(manifest) in completed.stdout
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert [item["density"] for item in data["results"]] == [0.5, 1.0]
    assert all(item["dry_run"] for item in data["results"])
