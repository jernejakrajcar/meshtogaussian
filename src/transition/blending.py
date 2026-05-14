from __future__ import annotations

from dataclasses import dataclass


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class TransitionWeights:
    mesh: float
    gaussian_lods: dict[str, float]

    def total(self) -> float:
        return self.mesh + sum(self.gaussian_lods.values())


class LODTransitionController:
    def __init__(self, config: dict):
        self.mesh_fade_start = float(config.get("mesh_fade_start", 3.6))
        self.mesh_fade_end = float(config.get("mesh_fade_end", 2.2))
        self.lod_ranges = {
            str(name): (float(value[0]), float(value[1]))
            for name, value in config.get("lod_ranges", {}).items()
        }

    def weights(self, distance: float) -> TransitionWeights:
        mesh_weight = 1.0 - smoothstep(self.mesh_fade_start, self.mesh_fade_end, distance)
        raw: dict[str, float] = {}
        for name, (far, near) in self.lod_ranges.items():
            enters = smoothstep(far, near, distance)
            exits = smoothstep(near, near * 0.72, distance) if near > 0.0 else 0.0
            raw[name] = max(0.0, enters * (1.0 - exits))

        gaussian_budget = max(0.0, 1.0 - mesh_weight)
        raw_total = sum(raw.values())
        if raw_total <= 1.0e-8:
            lod_weights = {name: 0.0 for name in self.lod_ranges}
            if gaussian_budget > 0.0 and self.lod_ranges:
                # Very close cameras can be past every finite fade band; keep
                # the highest LOD active instead of letting the image go dark.
                closest_lod = max(self.lod_ranges, key=self._lod_sort_key)
                lod_weights[closest_lod] = gaussian_budget
        else:
            lod_weights = {name: gaussian_budget * value / raw_total for name, value in raw.items()}

        total = mesh_weight + sum(lod_weights.values())
        if total > 1.0e-8:
            # Normalize after all fades so the final cross-fade is energy stable.
            mesh_weight /= total
            lod_weights = {name: value / total for name, value in lod_weights.items()}
        return TransitionWeights(mesh=mesh_weight, gaussian_lods=lod_weights)

    @staticmethod
    def _lod_sort_key(name: str) -> tuple[int, int | str]:
        return (1, int(name)) if name.isdigit() else (0, name)
