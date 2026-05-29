"""Majhen popravek za pycolmap SceneManager v lokalnem okolju.

Podporna/eksperimentalna datoteka: pomaga obiti neujemanja v
nameščeni knjižnici, da lahko ostali del pipeline-a bere COLMAP strukturo.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pycolmap.scene_manager as scene_manager


def main() -> None:
    path = Path(inspect.getsourcefile(scene_manager) or "")
    if not path.exists():
        raise RuntimeError("Could not locate pycolmap.scene_manager source file.")

    text = path.read_text(encoding="utf-8")
    replacements = {
        "Quaternion(np.array(map(float, data[1:5])))": "Quaternion(np.array(list(map(float, data[1:5]))))",
        "np.array(map(float, data[5:8]))": "np.array(list(map(float, data[5:8])))",
        "[map(float, data[::3]), map(float, data[1::3])]": "[list(map(float, data[::3])), list(map(float, data[1::3]))]",
        "np.array(map(np.uint64, data[2::3]))": "np.array(list(map(np.uint64, data[2::3])))",
        "self.points3D.append(map(np.float64, data[1:4]))": "self.points3D.append(list(map(np.float64, data[1:4])))",
        "self.point3D_colors.append(map(np.uint8, data[4:7]))": "self.point3D_colors.append(list(map(np.uint8, data[4:7])))",
        "np.array(map(np.uint32, data[8:])).reshape(-1, 2)": "np.array(list(map(np.uint32, data[8:]))).reshape(-1, 2)",
    }
    changed = False
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
            changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"Patched {path}")
    else:
        print(f"No patch needed for {path}")


if __name__ == "__main__":
    main()
