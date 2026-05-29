"""Izbira naprave za moznost cude/cpu ali po moznosti se dodati za amd??

Modul odloči, ali se uporablja CUDA ali CPU, in vrne kratek opis
naprave, da ostala koda ne ponavlja istega preverjanja.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    torch_device: Any
    backend: str
    description: str


class DeviceManager:
    """Centralized device selection for CPU, CUDA, and optional DirectML."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def resolve(self) -> DeviceInfo:
        preferred = str(self.config.get("preferred", "auto")).lower()
        allow_directml = bool(self.config.get("allow_directml", True))

        # CUDA ima prednost, ker je glavni cilj hitrejse renderiranje/trening z
        # PyTorchom, kadar je NVIDIA GPU na voljo.
        if preferred in {"auto", "cuda"} and torch.cuda.is_available():
            index = torch.cuda.current_device()
            return DeviceInfo(
                name="cuda",
                torch_device=torch.device("cuda"),
                backend="cuda",
                description=torch.cuda.get_device_name(index),
            )

        # DirectML je opcijski Windows fallback za racunalnike brez CUDA. V auto
        # nacinu ga poskusimo sele po CUDA, ker ni primarni cilj projekta.
        if preferred in {"directml", "dml", "auto"} and allow_directml:
            dml = self._try_directml()
            if dml is not None and preferred in {"directml", "dml"}:
                return dml
            if dml is not None and preferred == "auto":
                return dml

        # CPU fallback ohrani teste in majhne demo primere tudi brez GPU-ja.
        return DeviceInfo(
            name="cpu",
            torch_device=torch.device("cpu"),
            backend="cpu",
            description="CPU fallback",
        )

    @staticmethod
    def _try_directml() -> DeviceInfo | None:
        try:
            import torch_directml  # type: ignore
        except Exception:
            return None

        try:
            device = torch_directml.device()
        except Exception:
            return None

        return DeviceInfo(
            name="directml",
            torch_device=device,
            backend="directml",
            description="DirectML device via torch-directml",
        )
