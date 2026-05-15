from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import ensure_dir


SUPPORTED_MESH2SPLAT_INPUTS = {".obj", ".ply", ".gltf", ".glb"}


@dataclass(frozen=True)
class Mesh2SplatConfig:
    executable: Path
    output_dir: Path = Path("data/trained_gaussians/mesh2splat")
    glb_cache_dir: Path = Path("data/converted_glb")
    working_dir: Path | None = None
    density: float = 1.0
    timeout_seconds: int = 300
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Mesh2SplatConfig":
        cfg = data or {}
        executable = cfg.get("executable") or "Mesh2Splat.exe"
        working_dir = cfg.get("working_dir")
        return cls(
            executable=Path(executable),
            output_dir=Path(cfg.get("output_dir", "data/trained_gaussians/mesh2splat")),
            glb_cache_dir=Path(cfg.get("glb_cache_dir", "data/converted_glb")),
            working_dir=Path(working_dir) if working_dir else None,
            density=float(cfg.get("density", 1.0)),
            timeout_seconds=int(cfg.get("timeout_seconds", 300)),
            extra_args=[str(arg) for arg in cfg.get("extra_args", [])],
        )


@dataclass(frozen=True)
class Mesh2SplatResult:
    input_mesh: Path
    glb_mesh: Path
    output_ply: Path
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    manifest_path: Path


def convert_mesh_to_glb(mesh_path: str | Path, cache_dir: str | Path) -> Path:
    source = Path(mesh_path)
    if source.suffix.lower() == ".glb":
        return source
    if source.suffix.lower() not in SUPPORTED_MESH2SPLAT_INPUTS:
        raise ValueError(f"Unsupported Mesh2Splat input extension: {source.suffix}")

    try:
        import trimesh  # type: ignore
    except Exception as exc:
        raise RuntimeError("Converting meshes to .glb requires trimesh from requirements.txt.") from exc

    target_dir = ensure_dir(Path(cache_dir) / source.stem)
    target = target_dir / f"{source.stem}.glb"
    loaded = trimesh.load(source, force="scene")
    loaded.export(target)
    return target


def build_mesh2splat_command(
    config: Mesh2SplatConfig,
    input_glb: str | Path,
    output_ply: str | Path,
    density: float | None = None,
) -> list[str]:
    return [
        str(config.executable),
        "--headless",
        "--input",
        str(Path(input_glb)),
        "--output",
        str(Path(output_ply)),
        "--density",
        str(float(config.density if density is None else density)),
        "--quit",
        *config.extra_args,
    ]


def check_mesh2splat_environment(config: Mesh2SplatConfig) -> dict[str, Any]:
    problems: list[str] = []
    executable_found = config.executable.exists() or shutil.which(str(config.executable)) is not None
    if not executable_found:
        problems.append(f"Mesh2Splat executable was not found: {config.executable}")
    if config.working_dir is not None and not config.working_dir.exists():
        problems.append(f"Mesh2Splat working directory was not found: {config.working_dir}")
    try:
        ensure_dir(config.output_dir)
        ensure_dir(config.glb_cache_dir)
    except Exception as exc:
        problems.append(f"Could not create Mesh2Splat output/cache directories: {exc}")
    return {
        "ok": not problems,
        "executable": str(config.executable),
        "working_dir": str(config.working_dir) if config.working_dir else None,
        "output_dir": str(config.output_dir),
        "glb_cache_dir": str(config.glb_cache_dir),
        "problems": problems,
    }


def convert_mesh_to_gaussians(
    mesh_path: str | Path,
    config: Mesh2SplatConfig,
    density: float | None = None,
) -> Mesh2SplatResult:
    source = Path(mesh_path)
    if not source.exists():
        raise FileNotFoundError(f"Input mesh does not exist: {source}")
    status = check_mesh2splat_environment(config)
    if status["problems"]:
        raise RuntimeError("; ".join(status["problems"]))

    glb_path = convert_mesh_to_glb(source, config.glb_cache_dir)
    result_dir = ensure_dir(config.output_dir / source.stem)
    output_ply = result_dir / f"{source.stem}_mesh2splat.ply"
    command = build_mesh2splat_command(config, glb_path, output_ply, density=density)

    try:
        completed = subprocess.run(
            command,
            cwd=config.working_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Mesh2Splat timed out after {config.timeout_seconds} seconds.") from exc

    if completed.returncode != 0:
        _write_manifest(
            result_dir,
            source,
            glb_path,
            output_ply,
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )
        raise RuntimeError(
            "Mesh2Splat conversion failed "
            f"with exit code {completed.returncode}: {completed.stderr or completed.stdout}"
        )

    output_ply = _resolve_output_ply(result_dir, output_ply, timeout_seconds=min(config.timeout_seconds, 30))
    manifest_path = _write_manifest(
        result_dir,
        source,
        glb_path,
        output_ply,
        command,
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )
    return Mesh2SplatResult(
        input_mesh=source,
        glb_mesh=glb_path,
        output_ply=output_ply,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        manifest_path=manifest_path,
    )


def _resolve_output_ply(result_dir: Path, expected: Path, timeout_seconds: int) -> Path:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() <= deadline:
        if expected.exists():
            return expected
        candidates = sorted(result_dir.rglob("*.ply"), key=lambda path: path.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
        time.sleep(0.25)
    raise FileNotFoundError(
        "Mesh2Splat finished but no .ply was found. "
        "If the release executable has no headless export mode, add --headless/--input/--output support in Mesh2Splat."
    )


def _write_manifest(
    result_dir: Path,
    input_mesh: Path,
    glb_mesh: Path,
    output_ply: Path,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> Path:
    manifest = {
        "tool": "mesh2splat",
        "input_mesh": str(input_mesh),
        "glb_mesh": str(glb_mesh),
        "output_ply": str(output_ply),
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    path = result_dir / "mesh2splat_command.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
