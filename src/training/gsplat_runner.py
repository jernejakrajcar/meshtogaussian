from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def check_cuda_training_environment(gsplat_repo: str | Path | None = None) -> dict[str, Any]:
    status: dict[str, Any] = {"cuda_available": False, "gsplat_repo_ok": None, "problems": []}
    try:
        import torch
    except Exception as exc:
        status["problems"].append(f"PyTorch import failed: {exc}")
        return status

    status["torch_version"] = torch.__version__
    status["cuda_available"] = bool(torch.cuda.is_available())
    status["cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        status["cuda_version"] = torch.version.cuda
        status["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    else:
        status["problems"].append("torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build.")

    if gsplat_repo is not None:
        repo = Path(gsplat_repo)
        trainer = repo / "examples" / "simple_trainer.py"
        status["gsplat_repo"] = str(repo)
        status["gsplat_repo_ok"] = bool(trainer.exists())
        if not trainer.exists():
            status["problems"].append(f"Missing gsplat trainer script: {trainer}")
    return status


@dataclass(frozen=True)
class GsplatCommand:
    argv: list[str]
    cwd: Path
    result_dir: Path

    def as_shell_string(self) -> str:
        return " ".join(f'"{part}"' if " " in part else part for part in self.argv)


def build_gsplat_command(
    gsplat_repo: str | Path,
    data_dir: str | Path,
    result_dir: str | Path,
    python_executable: str | None = None,
    steps: int = 3000,
    data_factor: int = 1,
    extra_args: list[str] | None = None,
) -> GsplatCommand:
    repo = Path(gsplat_repo)
    script = repo / "examples" / "simple_trainer.py"
    py = python_executable or sys.executable
    argv = [
        py,
        str(script),
        "default",
        "--data_dir",
        str(Path(data_dir)),
        "--data_factor",
        str(data_factor),
        "--result_dir",
        str(Path(result_dir)),
        "--max_steps",
        str(steps),
    ]
    if extra_args:
        argv.extend(str(arg) for arg in extra_args)
    return GsplatCommand(argv=argv, cwd=repo, result_dir=Path(result_dir))


def run_gsplat_training(command: GsplatCommand, execute: bool = False) -> dict:
    command.result_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "command": command.argv,
        "cwd": str(command.cwd),
        "result_dir": str(command.result_dir),
        "executed": execute,
    }
    if execute:
        completed = subprocess.run(command.argv, cwd=command.cwd, check=False)
        summary["returncode"] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(f"gsplat training failed with exit code {completed.returncode}")
    (command.result_dir / "gsplat_command.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def find_latest_trained_ply(result_dir: str | Path) -> Path | None:
    candidates = sorted(Path(result_dir).rglob("*.ply"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None
