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
