from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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
