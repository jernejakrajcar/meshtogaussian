"""Testi za majhne JavaScript funkcije v spletni strani

S temi testi preverimo razvrščanje splattov po globini in formule zapostopno razkrivanje
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_depth_sorted_order_is_back_to_front() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = """
const { depthSortedOrder } = await import('./web/splat_sort.js');
const xyz = [[0, 0, -2], [0, 0, -5], [0, 0, -1]];
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
console.log(JSON.stringify(Array.from(depthSortedOrder(xyz, identity))));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [1, 0, 2]


def test_depth_sort_and_reorder_support_flat_float32_lods() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = """
const { depthSortedOrder, reorderByOrder } = await import('./web/splat_sort.js');
const xyz = new Float32Array([0, 0, -2, 0, 0, -5, 0, 0, -1]);
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const order = depthSortedOrder(xyz, identity);
console.log(JSON.stringify({ order: Array.from(order), xyz: Array.from(reorderByOrder(xyz, order, 3)) }));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "order": [1, 0, 2],
        "xyz": [0, 0, -5, 0, 0, -2, 0, 0, -1],
    }


def test_graphdeco_quaternion_reorders_for_shader() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = """
const { graphDecoToShaderQuat } = await import('./web/raw_gaussian_renderer.js');
console.log(JSON.stringify(graphDecoToShaderQuat([1, 2, 3, 4])));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [2, 3, 4, 1]


def test_detail_reveal_alpha_adds_only_new_rank_band() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = """
const { detailRevealAlpha } = await import('./web/raw_gaussian_renderer.js');
const values = [99999, 100000, 400000, 843145, 843146].map(
  (rank) => detailRevealAlpha(rank, 100000, 843146, 0.8),
);
console.log(JSON.stringify(values));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [1, 0.8, 0.8, 0.8, 0]
