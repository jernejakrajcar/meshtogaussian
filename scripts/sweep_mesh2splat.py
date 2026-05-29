"""Eksperimentalni sweep nastavitev za Mesh2Splat.

Skripta požene več gostot oziroma skal, zbere manifest in pomaga izbrati LOD
nivoje, ki so primerni za primerjavo v vizualizarju.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conversion.mesh2splat_runner import (
    Mesh2SplatConfig,
    build_mesh2splat_command,
    check_mesh2splat_environment,
    convert_mesh_to_glb,
)
from src.core.config import ensure_dir, load_config
from src.gaussian.coverage import analyze_gaussian_coverage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Mesh2Splat density/scale sweep and write coverage statistics for choosing LOD exports."
    )
    parser.add_argument("--mesh", required=True, help="Input OBJ/PLY/GLTF/GLB mesh.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--densities", default="0.5,1.0,1.5,2.0,3.0", help="Comma-separated Mesh2Splat density values.")
    parser.add_argument("--scale-values", default="", help="Optional comma-separated Gaussian scale values.")
    parser.add_argument("--scale-arg", default="--GaussianScale", help="Mesh2Splat CLI argument used for Gaussian scale.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional Mesh2Splat CLI argument. Repeat as needed.")
    parser.add_argument("--output-dir", default="data/mesh2splats")
    parser.add_argument("--sample-count", type=int, default=80000)
    parser.add_argument("--dry-run", action="store_true", help="Only write the commands that would be executed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_config = Mesh2SplatConfig.from_dict(cfg.get("mesh2splat", {}))
    status = check_mesh2splat_environment(base_config)
    # Pri dry-run dovolimo manjkajoce okolje, ker zelimo samo zapisati ukaze za
    # pregled; pri pravem zagonu pa bi to takoj padlo.
    if status["problems"] and not args.dry_run:
        raise SystemExit("; ".join(status["problems"]))

    mesh = Path(args.mesh)
    output_dir = ensure_dir(args.output_dir)
    sweep_dir = ensure_dir(output_dir / f"{mesh.stem}_sweep")
    glb = convert_mesh_to_glb(mesh, base_config.glb_cache_dir) if not args.dry_run else mesh
    densities = _float_list(args.densities)
    scale_values = _float_list(args.scale_values) or [None]

    results = []
    for density in densities:
        for scale in scale_values:
            run_name = _run_name(mesh.stem, density, scale)
            temp_ply = sweep_dir / f"{run_name}.ply"
            extra_args = [*base_config.extra_args, *args.extra_arg]
            # Scale parameter je opcijski, ker ga ne podpirajo nujno vse verzije
            # Mesh2Splat CLI-ja.
            if scale is not None:
                extra_args.extend([args.scale_arg, str(scale)])
            config = replace(base_config, output_dir=sweep_dir, extra_args=extra_args)
            command = build_mesh2splat_command(config, glb, temp_ply, density=density)
            entry = {
                "density": density,
                "gaussian_scale": scale,
                "temporary_output": str(temp_ply),
                "command": command,
            }
            if args.dry_run:
                entry["dry_run"] = True
                results.append(entry)
                continue

            completed = subprocess.run(
                command,
                cwd=config.working_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
            )
            entry["returncode"] = completed.returncode
            entry["stdout"] = completed.stdout
            entry["stderr"] = completed.stderr
            # Ne prekinem celotnega sweepa pri eni neuspeli kombinaciji; rezultat
            # shranim v manifest in grem na naslednji poskus.
            if completed.returncode != 0:
                entry["error"] = completed.stderr or completed.stdout
                results.append(entry)
                continue
            if not temp_ply.exists():
                candidates = sorted(sweep_dir.rglob("*.ply"), key=lambda path: path.stat().st_mtime, reverse=True)
                if not candidates:
                    entry["error"] = "Mesh2Splat completed but no PLY was found."
                    results.append(entry)
                    continue
                temp_ply = candidates[0]

            stats = analyze_gaussian_coverage(temp_ply, sample_count=args.sample_count)
            final_ply = output_dir / f"{mesh.stem}-{stats.count}.ply"
            # Koncno ime vsebuje dejansko stevilo splattov, ker je to bolj
            # uporabno za LOD izbiro kot vhodna gostota.
            if temp_ply.resolve() != final_ply.resolve():
                shutil.copy2(temp_ply, final_ply)
            entry["output_ply"] = str(final_ply)
            entry["stats"] = stats.to_dict()
            results.append(entry)

    manifest = {
        "mesh": str(mesh),
        "output_dir": str(output_dir),
        "densities": densities,
        "scale_values": scale_values,
        "recommended_lods": _recommended_lods(results),
        "results": results,
    }
    manifest_path = output_dir / f"{mesh.stem}_mesh2splat_sweep.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Mesh2Splat sweep manifest: {manifest_path}")
    for item in manifest["recommended_lods"]:
        print(f"recommended LOD: {item}")


def _float_list(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _run_name(stem: str, density: float, scale: float | None) -> str:
    density_label = str(density).replace(".", "p")
    if scale is None:
        return f"{stem}-density-{density_label}"
    scale_label = str(scale).replace(".", "p")
    return f"{stem}-density-{density_label}-scale-{scale_label}"


def _recommended_lods(results: list[dict], max_levels: int = 7) -> list[str]:
    successful = [
        item
        for item in results
        if item.get("output_ply") and item.get("stats", {}).get("count", 0) > 0
    ]
    successful.sort(key=lambda item: int(item["stats"]["count"]))
    # Ce je uspesnih rezultatov malo, jih obdrzim vse; sicer vzamem enakomerno
    # razporejene gostote, da LOD set ni prevelik.
    if len(successful) <= max_levels:
        return [item["output_ply"] for item in successful]
    positions = sorted({round(i * (len(successful) - 1) / (max_levels - 1)) for i in range(max_levels)})
    return [successful[index]["output_ply"] for index in positions]


if __name__ == "__main__":
    main()
