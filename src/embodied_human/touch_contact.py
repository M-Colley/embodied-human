"""Body contact maps on an existing MuJoCo model, using adapted MIMo sampling.

Locations are in metres and forces in newtons when the model uses SI units.
This pathway reports physical contacts and preserves the existing vibration
psychophysics; it supplies no force to indentation or perception conversion.
MuJoCo is imported only when binding a model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ._vendor.mimo_touch import (
    box_points,
    cylinder_points,
    scale_linear,
    sphere_points,
)
from .core import Source, register
from .touch import TouchParameters, VibrotactileTouch

BodyKey = str | int
_MIMO = (
    "https://github.com/trieschlab/MIMo/blob/"
    "040b0ae4914cbfb26afdf830aa81775b90922f3f/mimoTouch/"
)


@dataclass(frozen=True)
class ContactParameters(TouchParameters):
    """Explicit sensor map protocol choices, not fitted human sensitivity."""

    sensor_spacing: float = 0.02
    spread_radius: float = 0.04
    response: str = "spread_linear"
    max_sensors_per_body: int = 20000

    SOURCES: ClassVar = {
        **TouchParameters.SOURCES,
        "sensor_spacing": Source(
            "Configurable geometric sampling convention",
            _MIMO + "sensorpoints.py",
            "assumed",
            "metres; not a measured receptor density or discrimination threshold",
        ),
        "spread_radius": Source(
            "MIMo linear spatial response, protocol radius selected here",
            _MIMO + "touch.py",
            "assumed",
            "metres; default twice the default spacing, not a skin mechanics fit",
        ),
        "response": Source(
            "MIMo nearest or linear force allocation",
            _MIMO + "touch.py",
            "assumed",
            "normalised spatial weights conserve all three force components",
        ),
        "max_sensors_per_body": Source(
            "Sensor allocation limit",
            None,
            "assumed",
            "runtime allocation limit, not a physiological parameter",
        ),
    }


@dataclass(frozen=True)
class SensorMap:
    """Body frame positions in metres and optional outward unit normals."""

    positions: np.ndarray
    normals: np.ndarray | None = None

    def __post_init__(self) -> None:
        positions = np.array(self.positions, dtype=float, copy=True)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or len(positions) == 0
            or not np.isfinite(positions).all()
        ):
            raise ValueError("sensor positions must be a finite nonempty (n, 3) array")
        positions.flags.writeable = False
        object.__setattr__(self, "positions", positions)
        if self.normals is not None:
            normals = np.array(self.normals, dtype=float, copy=True)
            if normals.shape != positions.shape or not np.isfinite(normals).all():
                raise ValueError("sensor normals must be finite and match positions")
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            if np.any(lengths == 0) or not np.isfinite(lengths).all():
                raise ValueError("sensor normals must have nonzero length")
            normals /= lengths
            normals.flags.writeable = False
            object.__setattr__(self, "normals", normals)


@dataclass(frozen=True)
class BodyContact:
    """One solved rigid contact viewed from one sensing body.

    ``position`` is MuJoCo's contact midpoint in the body frame. ``force`` and
    intrinsic ``torque`` act on that body, expressed in the same frame. Contact
    torques are not distributed to force sensors. The outward normal points
    from this body towards the other contacting geom.
    """

    contact_index: int
    body_id: int
    other_body_id: int
    geom_id: int
    other_geom_id: int
    position: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    outward_normal: np.ndarray
    world_position: np.ndarray
    world_force: np.ndarray
    nearest_fallback: bool


@dataclass(frozen=True)
class ContactReadout:
    """Detached contact and force map snapshot at the current simulation time."""

    time: float
    contacts: tuple[BodyContact, ...]
    sensor_positions: dict[BodyKey, np.ndarray]
    sensor_forces: dict[BodyKey, np.ndarray]

    @property
    def fallback_count(self) -> int:
        """Allocations that required nearest fallback outside the configured map."""
        return sum(contact.nearest_fallback for contact in self.contacts)


@register("touch", "contact_map")
class ContactMapTouch(VibrotactileTouch):
    """Read physical contacts on the existing body without changing its model.

    ``bind`` keeps references to the caller's model and data. ``sense`` only
    reads them, so the caller must first step the engine or run ``mj_forward``.
    Human reset changes neither the engine state nor the configured sensor map.
    """

    params: ContactParameters

    def __init__(self, **params: Any) -> None:
        super().__init__(
            **{
                key: value
                for key, value in params.items()
                if key in TouchParameters.SOURCES
            }
        )
        self.params = ContactParameters(**params)
        p = self.params
        if not math.isfinite(p.sensor_spacing) or p.sensor_spacing <= 0:
            raise ValueError("sensor_spacing must be finite and positive")
        if not math.isfinite(p.spread_radius) or p.spread_radius <= 0:
            raise ValueError("spread_radius must be finite and positive")
        if p.response not in ("nearest", "spread_linear"):
            raise ValueError("response must be 'nearest' or 'spread_linear'")
        if (
            isinstance(p.max_sensors_per_body, bool)
            or not isinstance(p.max_sensors_per_body, int)
            or p.max_sensors_per_body < 1
        ):
            raise ValueError("max_sensors_per_body must be a positive integer")
        self._model: Any = None
        self._data: Any = None
        self._mujoco: Any = None
        self._maps: dict[BodyKey, SensorMap] = {}
        self._keys: dict[int, BodyKey] = {}

    def bind(
        self,
        model: Any,
        data: Any,
        sensors: Mapping[BodyKey, SensorMap | Sequence[Sequence[float]] | np.ndarray]
        | None = None,
        *,
        bodies: Sequence[BodyKey] | None = None,
    ) -> None:
        """Bind existing MuJoCo objects with explicit or generated sensor maps.

        Map keys are body names or numeric IDs and are preserved in readouts.
        With no explicit map, sample the selected bodies' contact enabled geoms.
        With no selection, sample all bodies with contact enabled geoms except
        the world body. Primitive geoms and existing mesh vertices are supported;
        planes and height fields require an explicit map with a finite extent.
        """
        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise ImportError(
                "contact sensing requires MuJoCo: python -m pip install 'mujoco>=3.7'"
            ) from exc
        if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
            raise TypeError("bind requires an MjModel and its MjData")
        if data.model is not model:
            raise ValueError("data must belong to the supplied model")
        if sensors is not None and bodies is not None:
            raise ValueError("supply sensors or bodies, not both")
        if isinstance(bodies, (str, bytes)):
            raise ValueError("bodies must be a sequence of body names or IDs")
        if sensors is None:
            selected: Sequence[BodyKey] = (
                bodies
                if bodies is not None
                else [
                    model.body(body_id).name or body_id
                    for body_id in range(1, model.nbody)
                    if self._body_geoms(model, body_id)
                ]
            )
            generated = {}
            for key in selected:
                body_id = self._body_id(model, key)
                generated[key] = self._sample_body(model, mujoco, body_id)
            sensors = generated
        if not sensors:
            raise ValueError("at least one sensing body is required")
        maps, keys = {}, {}
        for key, sensor_map in sensors.items():
            body_id = self._body_id(model, key)
            if body_id in keys:
                raise ValueError("each body may have only one sensor map")
            mapping = (
                sensor_map
                if isinstance(sensor_map, SensorMap)
                else SensorMap(np.asarray(sensor_map))
            )
            if len(mapping.positions) > self.params.max_sensors_per_body:
                raise ValueError("sensor map exceeds max_sensors_per_body")
            maps[key], keys[body_id] = mapping, key
        self._model, self._data, self._mujoco = model, data, mujoco
        self._maps, self._keys = maps, keys

    @staticmethod
    def _body_id(model: Any, key: BodyKey) -> int:
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise ValueError("body keys must be names or integer IDs")
        if isinstance(key, int) and not 0 <= key < model.nbody:
            raise ValueError(f"unknown body {key!r}")
        try:
            return int(model.body(key).id)
        except (KeyError, IndexError) as exc:
            raise ValueError(f"unknown body {key!r}") from exc

    @staticmethod
    def _body_geoms(model: Any, body_id: int) -> list[int]:
        return [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == body_id
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]

    def _sample_body(self, model: Any, mujoco: Any, body_id: int) -> SensorMap:
        positions, normals = [], []
        has_normals = True
        for geom_id in self._body_geoms(model, body_id):
            points, directions = self._sample_geom(model, mujoco, geom_id)
            flat_rotation = np.empty(9)
            mujoco.mju_quat2Mat(flat_rotation, model.geom_quat[geom_id])
            rotation = flat_rotation.reshape(3, 3)
            positions.append(points @ rotation.T + model.geom_pos[geom_id])
            if directions is None:
                has_normals = False
            else:
                normals.append(directions @ rotation.T)
            if sum(len(p) for p in positions) > self.params.max_sensors_per_body:
                raise ValueError("sensor map exceeds max_sensors_per_body")
        if not positions:
            raise ValueError(f"body {body_id} has no contact enabled geoms")
        points = np.concatenate(positions)
        points, indices = np.unique(points, axis=0, return_index=True)
        directions = np.concatenate(normals)[indices] if has_normals else None
        return SensorMap(points, directions)

    def _sample_geom(
        self, model: Any, mujoco: Any, geom_id: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        kind, size = model.geom_type[geom_id], model.geom_size[geom_id]
        spacing, limit = self.params.sensor_spacing, self.params.max_sensors_per_body
        geom = mujoco.mjtGeom
        if kind == geom.mjGEOM_BOX:
            return box_points(spacing, size, limit)
        if kind == geom.mjGEOM_SPHERE:
            return sphere_points(spacing, np.repeat(size[0], 3), limit)
        if kind == geom.mjGEOM_ELLIPSOID:
            return sphere_points(spacing, size, limit)
        if kind in (geom.mjGEOM_CAPSULE, geom.mjGEOM_CYLINDER):
            return cylinder_points(
                spacing,
                2 * size[1],
                size[0],
                limit,
                capsule=kind == geom.mjGEOM_CAPSULE,
            )
        if kind == geom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            if count > limit:
                raise ValueError("mesh vertex map exceeds max_sensors_per_body")
            # Use existing compiled mesh vertices. No resampling or replacement
            # of the model's geometry, and no invented anatomical sensor normals.
            return np.array(model.mesh_vert[start : start + count], dtype=float), None
        raise ValueError(
            f"geom {geom_id} needs an explicit SensorMap; automatic sampling "
            "supports boxes, spheres, ellipsoids, capsules, cylinders and meshes"
        )

    def sense(self) -> ContactReadout:
        """Read solved rigid contacts and conserve each body's contact forces."""
        if self._model is None:
            raise RuntimeError("bind a MuJoCo model and data before calling sense")
        model, data = self._model, self._data
        outputs = {key: np.zeros_like(m.positions) for key, m in self._maps.items()}
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if min(geom1, geom2) < 0 or contact.efc_address < 0:
                continue
            body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
            if body1 not in self._keys and body2 not in self._keys:
                continue
            wrench = np.empty(6)
            self._mujoco.mj_contactForce(model, data, index, wrench)
            frame = np.asarray(contact.frame).reshape(3, 3)
            for body_id, other_id, geom_id, other_geom, sign in (
                (body1, body2, geom1, geom2, -1),
                (body2, body1, geom2, geom1, 1),
            ):
                if body_id not in self._keys:
                    continue
                rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
                world_force = sign * frame.T @ wrench[:3]
                position = rotation.T @ (contact.pos - data.xpos[body_id])
                force = rotation.T @ world_force
                normal = rotation.T @ (-sign * frame[0])
                key = self._keys[body_id]
                distributed, fallback = self._allocate(
                    self._maps[key], position, normal, force
                )
                outputs[key] += distributed
                contacts.append(
                    BodyContact(
                        index,
                        body_id,
                        other_id,
                        geom_id,
                        other_geom,
                        np.array(position),
                        np.array(force),
                        rotation.T @ (sign * frame.T @ wrench[3:]),
                        np.array(normal),
                        np.array(contact.pos),
                        np.array(world_force),
                        fallback,
                    )
                )
        return ContactReadout(
            float(data.time),
            tuple(contacts),
            {key: m.positions.copy() for key, m in self._maps.items()},
            outputs,
        )

    def _allocate(
        self,
        mapping: SensorMap,
        position: np.ndarray,
        normal: np.ndarray,
        force: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        distances = np.linalg.norm(mapping.positions - position, axis=1)
        eligible = np.ones(len(distances), dtype=bool)
        if mapping.normals is not None:
            facing = mapping.normals @ normal > 0
            if np.any(facing):
                eligible = facing
            else:
                output = np.zeros_like(mapping.positions)
                output[np.argmin(distances)] = force
                return output, True
        indices = np.flatnonzero(eligible)
        output = np.zeros_like(mapping.positions)
        if self.params.response == "nearest":
            output[indices[np.argmin(distances[indices])]] = force
            return output, False
        # Reuse MIMo's spatial response, but normalise scalar weights rather than
        # one force component. Pure tangential forces must also be conserved.
        weights = scale_linear(
            np.ones(1), distances[indices], self.params.spread_radius
        )[:, 0]
        total = float(weights.sum())
        if total <= 0:
            output[indices[np.argmin(distances[indices])]] = force
            return output, True
        output[indices] = weights[:, None] / total * force
        return output, False
