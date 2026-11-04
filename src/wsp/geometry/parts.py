"""Parametric machine-part primitives and their analytic signed distance fields (SDF).

All public dimensions are in **millimetres**. Networks work in a normalised frame where
``x_norm = x_mm / MM_PER_UNIT`` so that the whole workspace ``[-50 mm, 50 mm]^3`` maps to
``[-1, 1]^3``. The SDF convention is *negative inside material*, positive outside.

The CSG combinations used here (``max(a, -b)`` for subtraction) give an exact zero level set
but only a bound on the distance away from it, which is the usual situation for CAD-style SDFs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import numpy as np

MM_PER_UNIT = 50.0
"""Physical size of one normalised unit. The workspace [-1, 1]^3 is 100 mm wide."""

Family = Literal["plate", "through_hole", "blind_hole", "counterbore", "flange"]
FAMILIES: tuple[Family, ...] = ("plate", "through_hole", "blind_hole", "counterbore", "flange")


@dataclass(frozen=True)
class HoleSpec:
    """A cylindrical hole drilled along -Z from the top face.

    ``depth=None`` means a through hole. ``cb_radius``/``cb_depth`` describe a counterbore
    (ザグリ) at the top of the hole.
    """

    cx: float
    cy: float
    radius: float
    depth: float | None = None
    cb_radius: float | None = None
    cb_depth: float | None = None

    @property
    def is_through(self) -> bool:
        return self.depth is None

    @property
    def has_counterbore(self) -> bool:
        return self.cb_radius is not None and self.cb_depth is not None


@dataclass(frozen=True)
class PartSpec:
    """A plate (``outline='rect'``) or flange disk (``outline='disk'``) with holes.

    For a disk, ``size_x == size_y`` is the outer diameter.
    """

    name: str
    outline: Literal["rect", "disk"]
    size_x: float
    size_y: float
    thickness: float
    z_bottom: float
    center_x: float = 0.0
    center_y: float = 0.0
    holes: tuple[HoleSpec, ...] = field(default_factory=tuple)

    @property
    def z_top(self) -> float:
        return self.z_bottom + self.thickness

    @property
    def family(self) -> Family:
        if self.outline == "disk":
            return "flange"
        if not self.holes:
            return "plate"
        main = self.holes[0]
        if main.has_counterbore:
            return "counterbore"
        if not main.is_through:
            return "blind_hole"
        return "through_hole"

    def bbox_mm(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.center_x - self.size_x / 2, self.center_y - self.size_y / 2, self.z_bottom])
        hi = np.array([self.center_x + self.size_x / 2, self.center_y + self.size_y / 2, self.z_top])
        return lo, hi

    def to_dict(self) -> dict:
        d = asdict(self)
        d["family"] = self.family
        return d

    @staticmethod
    def from_dict(d: dict) -> PartSpec:
        d = {k: v for k, v in d.items() if k != "family"}
        holes = tuple(HoleSpec(**h) for h in d.pop("holes", ()))
        return PartSpec(holes=holes, **d)

    def with_(self, **kwargs) -> PartSpec:
        return replace(self, **kwargs)


# ----------------------------------------------------------------------------------------------
# Analytic SDF primitives (millimetres in, millimetres out)
# ----------------------------------------------------------------------------------------------


def _box_sdf(p: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    q = np.abs(p - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


def _capped_cylinder_sdf(p: np.ndarray, cx: float, cy: float, r: float, z0: float, z1: float) -> np.ndarray:
    d_r = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r
    zc, hz = (z0 + z1) / 2, (z1 - z0) / 2
    d_z = np.abs(p[:, 2] - zc) - hz
    q = np.stack([d_r, d_z], axis=-1)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def _infinite_cylinder_sdf(p: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    return np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r


def part_sdf_mm(spec: PartSpec, p_mm: np.ndarray) -> np.ndarray:
    """Signed distance (mm) of ``spec`` at points ``p_mm`` with shape (N, 3)."""
    p = np.asarray(p_mm, dtype=np.float64).reshape(-1, 3)
    if spec.outline == "rect":
        center = np.array([spec.center_x, spec.center_y, spec.z_bottom + spec.thickness / 2])
        half = np.array([spec.size_x / 2, spec.size_y / 2, spec.thickness / 2])
        d = _box_sdf(p, center, half)
    else:
        d = _capped_cylinder_sdf(p, spec.center_x, spec.center_y, spec.size_x / 2, spec.z_bottom, spec.z_top)

    # Holes are cut slightly above the top face so the cut never leaves a skin.
    above = spec.z_top + 1.0e3
    for h in spec.holes:
        if h.is_through:
            cut = _infinite_cylinder_sdf(p, h.cx, h.cy, h.radius)
        else:
            assert h.depth is not None
            cut = _capped_cylinder_sdf(p, h.cx, h.cy, h.radius, spec.z_top - h.depth, above)
        d = np.maximum(d, -cut)
        if h.has_counterbore:
            assert h.cb_radius is not None and h.cb_depth is not None
            cb = _capped_cylinder_sdf(p, h.cx, h.cy, h.cb_radius, spec.z_top - h.cb_depth, above)
            d = np.maximum(d, -cb)
    return d


def part_sdf_norm(spec: PartSpec, p_norm: np.ndarray) -> np.ndarray:
    """Signed distance in normalised units at normalised points."""
    return part_sdf_mm(spec, np.asarray(p_norm) * MM_PER_UNIT) / MM_PER_UNIT


# ----------------------------------------------------------------------------------------------
# Random part generator (deterministic given the RNG)
# ----------------------------------------------------------------------------------------------


def random_part(family: Family, rng: np.random.Generator, name: str | None = None) -> PartSpec:
    """Sample a plausible part of the given family. Dimensions are rounded to 0.5 mm."""

    def r05(x: float) -> float:
        return float(np.round(x * 2) / 2)

    thickness = r05(rng.uniform(8.0, 14.0))
    z_bottom = r05(-thickness / 2 + rng.uniform(-3.0, 3.0))
    name = name or family

    if family == "flange":
        diameter = r05(rng.uniform(60.0, 80.0))
        bore = r05(rng.uniform(7.0, 12.0))
        return PartSpec(
            name=name,
            outline="disk",
            size_x=diameter,
            size_y=diameter,
            thickness=thickness,
            z_bottom=z_bottom,
            holes=(HoleSpec(0.0, 0.0, bore),),
        )

    size_x = r05(rng.uniform(60.0, 80.0))
    size_y = r05(rng.uniform(50.0, 75.0))
    if family == "plate":
        return PartSpec(name, "rect", size_x, size_y, thickness, z_bottom)

    radius = r05(rng.uniform(6.0, 11.0))
    margin = radius + 12.0
    cx = r05(rng.uniform(-size_x / 2 + margin, size_x / 2 - margin))
    cy = r05(rng.uniform(-size_y / 2 + margin, size_y / 2 - margin))
    if family == "through_hole":
        hole = HoleSpec(cx, cy, radius)
    elif family == "blind_hole":
        hole = HoleSpec(cx, cy, radius, depth=r05(thickness * rng.uniform(0.45, 0.65)))
    elif family == "counterbore":
        radius = r05(rng.uniform(5.0, 7.0))
        hole = HoleSpec(
            cx,
            cy,
            radius,
            cb_radius=r05(radius * rng.uniform(1.6, 1.9)),
            cb_depth=r05(thickness * rng.uniform(0.35, 0.5)),
        )
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(family)
    return PartSpec(name, "rect", size_x, size_y, thickness, z_bottom, holes=(hole,))


def make_part_set(n_per_family: int, seed: int) -> list[PartSpec]:
    rng = np.random.default_rng(seed)
    parts = []
    for fam in FAMILIES:
        for i in range(n_per_family):
            parts.append(random_part(fam, rng, name=f"{fam}_{i}"))
    return parts
