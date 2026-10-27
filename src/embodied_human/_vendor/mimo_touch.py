"""Adapted NumPy geometry and force scaling from MIMo, under MIT.

Source: trieschlab/MIMo at 040b0ae4914cbfb26afdf830aa81775b90922f3f,
mimoTouch/sensorpoints.py and mimoTouch/touch.py (scale_linear).
Copyright (c) 2025 Francisco M. López and the MIMo contributors.
Full copyright and permission notice: MIMO_LICENSE.txt in this directory.

Changes: retained only the point sampling and linear scaling operations, added
type annotations and size limits, and retained surface points at coarse spacing.
No Gymnasium, Trimesh, plotting or environment implementation is included.
"""

from __future__ import annotations

import math

import numpy as np


def scale_linear(force: np.ndarray, distances: np.ndarray, scale: float) -> np.ndarray:
    """MIMo linear force falloff, before force conserving normalization."""
    factor = (scale - distances) / scale
    factor[factor < 0] = 0
    return (force[:, np.newaxis] * factor).T


def _check_count(count: float, limit: int) -> None:
    if not math.isfinite(count) or count > limit:
        raise ValueError("sensor map exceeds max_sensors_per_body; increase spacing")


def sphere_points(
    spacing: float, radii: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """MIMo latitude ring sampling, extended to ellipsoid surface normals."""
    radius = float(np.max(radii))
    _check_count(8 + 4 * math.pi * (radius / spacing) ** 2, limit)
    points = []
    n_theta = max(2, int(math.ceil(math.pi * radius / spacing)))
    for i in range(n_theta):
        theta = (i + 0.5) * math.pi / n_theta
        n_phi = max(1, round(2 * math.pi * math.sin(theta) * radius / spacing))
        for j in range(n_phi):
            phi = 2 * math.pi * j / n_phi
            points.append(
                [
                    radii[0] * math.sin(theta) * math.cos(phi),
                    radii[1] * math.sin(theta) * math.sin(phi),
                    radii[2] * math.cos(theta),
                ]
            )
    positions = np.asarray(points, dtype=float)
    normals = positions / radii**2
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return positions, normals


def box_points(
    spacing: float, sizes: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """MIMo subdivision of the six box faces, without duplicate edge points."""
    counts = np.maximum(2, np.ceil(2 * sizes / spacing))
    _check_count(float(np.max(counts)), limit)
    divisions = counts.astype(np.int64)
    nx, ny, nz = [int(n) for n in divisions]
    _check_count(nx * ny * nz - (nx - 2) * (ny - 2) * (nz - 2), limit)
    x_coords, y_coords, z_coords = [
        np.linspace(-size, size, count) for size, count in zip(sizes, divisions)
    ]
    points = [
        (x, y, z)
        for x in (x_coords[0], x_coords[-1])
        for y in y_coords
        for z in z_coords
    ]
    points += [
        (x, y, z)
        for x in x_coords[1:-1]
        for y in (y_coords[0], y_coords[-1])
        for z in z_coords
    ]
    points += [
        (x, y, z)
        for x in x_coords[1:-1]
        for y in y_coords[1:-1]
        for z in (z_coords[0], z_coords[-1])
    ]
    positions = np.asarray(points, dtype=float)
    normals = np.where(np.isclose(np.abs(positions), sizes), np.sign(positions), 0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return positions, normals


def cylinder_points(
    spacing: float, length: float, radius: float, limit: int, *, capsule: bool
) -> tuple[np.ndarray, np.ndarray]:
    """MIMo cylindrical rings and concentric disc or hemisphere end caps."""
    _check_count(
        16 + (2 * math.pi * radius * length + 4 * math.pi * radius**2) / spacing**2,
        limit,
    )
    n_length = max(2, int(math.ceil(length / spacing)) + 1)
    n_circum = max(3, int(math.ceil(2 * math.pi * radius / spacing)))
    points, normals = [], []
    for i in range(n_length):
        z = i * length / (n_length - 1) - length / 2
        for j in range(n_circum):
            theta = 2 * math.pi * j / n_circum
            x, y = radius * math.cos(theta), radius * math.sin(theta)
            points.append([x, y, z])
            normals.append([x / radius, y / radius, 0])
    if capsule:
        n_theta = max(1, int(round(math.pi * radius / (2 * spacing))))
        for i in range(n_theta):
            theta = i * math.pi / (2 * n_theta)
            n_phi = max(1, round(2 * math.pi * math.sin(theta) * radius / spacing))
            for j in range(n_phi):
                phi = 2 * math.pi * j / n_phi
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z_normal = radius * math.cos(theta)
                for sign in (-1, 1):
                    points.append([x, y, sign * (z_normal + length / 2)])
                    normals.append([x / radius, y / radius, sign * z_normal / radius])
    else:
        n_caps = max(2, int(round(radius / spacing)) + 1)
        for cap_radius in np.linspace(0, radius, n_caps)[:-1]:
            n_cap = max(1, int(math.floor(2 * math.pi * cap_radius / spacing)))
            for j in range(n_cap):
                theta = 2 * math.pi * (j + 0.5) / n_cap
                x, y = cap_radius * math.cos(theta), cap_radius * math.sin(theta)
                for sign in (-1, 1):
                    points.append([x, y, sign * length / 2])
                    normals.append([0, 0, sign])
    _check_count(len(points), limit)
    return np.asarray(points), np.asarray(normals)
