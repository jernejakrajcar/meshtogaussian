from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan

import numpy as np


@dataclass(frozen=True)
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_degrees: float
    width: int
    height: int

    @property
    def view_matrix(self) -> np.ndarray:
        # Camera space follows the common OpenGL-style convention: the camera
        # looks down its negative Z axis, so visible points have z < 0.
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ self.eye
        return view

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        fy = 0.5 * self.height / tan(radians(self.fov_y_degrees) * 0.5)
        fx = fy
        cx = (self.width - 1) * 0.5
        cy = (self.height - 1) * 0.5
        return fx, fy, cx, cy

    def distance_to_origin(self) -> float:
        return float(np.linalg.norm(self.eye))


class CameraRig:
    @staticmethod
    def orbit(
        radius: float,
        elevations: list[float],
        azimuth_count: int,
        image_size: tuple[int, int],
        fov_y_degrees: float = 45.0,
    ) -> list[Camera]:
        width, height = image_size
        cameras: list[Camera] = []
        for elevation in elevations:
            elev = radians(elevation)
            for index in range(azimuth_count):
                az = radians(index * 360.0 / azimuth_count)
                eye = np.array(
                    [
                        radius * cos(elev) * sin(az),
                        radius * sin(elev),
                        radius * cos(elev) * cos(az),
                    ],
                    dtype=np.float32,
                )
                cameras.append(
                    Camera(
                        eye=eye,
                        target=np.zeros(3, dtype=np.float32),
                        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                        fov_y_degrees=fov_y_degrees,
                        width=width,
                        height=height,
                    )
                )
        return cameras

    @staticmethod
    def transition_path(
        far_radius: float,
        near_radius: float,
        frames: int,
        image_size: tuple[int, int],
        azimuth_degrees: float,
        elevation_degrees: float,
        fov_y_degrees: float = 45.0,
    ) -> list[Camera]:
        width, height = image_size
        az = radians(azimuth_degrees)
        elev = radians(elevation_degrees)
        cameras = []
        for t in np.linspace(0.0, 1.0, frames):
            smooth_t = t * t * (3.0 - 2.0 * t)
            radius = (1.0 - smooth_t) * far_radius + smooth_t * near_radius
            eye = np.array(
                [
                    radius * cos(elev) * sin(az),
                    radius * sin(elev),
                    radius * cos(elev) * cos(az),
                ],
                dtype=np.float32,
            )
            cameras.append(
                Camera(
                    eye=eye,
                    target=np.zeros(3, dtype=np.float32),
                    up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    fov_y_degrees=fov_y_degrees,
                    width=width,
                    height=height,
                )
            )
        return cameras
