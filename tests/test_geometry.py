"""Parametric parts, meshing and metrics."""

import numpy as np

from wsp.geometry.mesh import (
    chamfer_distance,
    extract_mesh,
    is_watertight,
    n_components,
    sample_surface,
    volumetric_iou,
)
from wsp.geometry.parts import FAMILIES, MM_PER_UNIT, HoleSpec, PartSpec, make_part_set, part_sdf_mm, random_part


def test_part_families_and_signs():
    rng = np.random.default_rng(0)
    for fam in FAMILIES:
        spec = random_part(fam, rng)
        assert spec.family == fam
        _, hi = spec.bbox_mm()
        outside = hi + 5.0
        assert part_sdf_mm(spec, outside[None])[0] > 0
    plate = PartSpec("p", "rect", 60, 40, 10, -5, holes=(HoleSpec(10, 0, 5),))
    assert part_sdf_mm(plate, np.array([[-20.0, 0, 0]]))[0] < 0  # material
    assert part_sdf_mm(plate, np.array([[10.0, 0, 0]]))[0] > 0  # inside the through hole


def test_make_part_set_is_deterministic():
    a, b = make_part_set(1, seed=3), make_part_set(1, seed=3)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]
    assert PartSpec.from_dict(a[1].to_dict()) == a[1]


def test_box_mesh_metrics():
    spec = PartSpec("box", "rect", 40, 30, 10, -5)
    fn = lambda p: part_sdf_mm(spec, p * MM_PER_UNIT) / MM_PER_UNIT  # noqa: E731
    lo, hi = spec.bbox_mm()
    mesh = extract_mesh(fn, (lo - 3) / MM_PER_UNIT, (hi + 3) / MM_PER_UNIT, 0.5 / MM_PER_UNIT)
    assert is_watertight(mesh)
    assert n_components(mesh) == 1
    vol_mm3 = mesh.volume() * MM_PER_UNIT**3
    assert abs(vol_mm3 - 40 * 30 * 10) / (40 * 30 * 10) < 0.02
    pts = sample_surface(mesh, 5000, np.random.default_rng(0))
    assert chamfer_distance(pts, pts)["chamfer"] == 0.0
    assert volumetric_iou(fn, fn, (lo - 3) / MM_PER_UNIT, (hi + 3) / MM_PER_UNIT, 1 / MM_PER_UNIT) == 1.0


def test_feature_weighted_sampling_prefers_walls():
    spec = PartSpec("p", "rect", 60, 60, 10, -5, holes=(HoleSpec(0, 0, 10),))
    fn = lambda p: part_sdf_mm(spec, p * MM_PER_UNIT) / MM_PER_UNIT  # noqa: E731
    lo, hi = spec.bbox_mm()
    mesh = extract_mesh(fn, (lo - 3) / MM_PER_UNIT, (hi + 3) / MM_PER_UNIT, 1 / MM_PER_UNIT)
    rng = np.random.default_rng(0)
    plain = sample_surface(mesh, 20000, rng)
    weighted = sample_surface(mesh, 20000, rng, feature_weight=16)

    def wall_fraction(p):
        r = np.hypot(p[:, 0], p[:, 1]) * MM_PER_UNIT
        return np.mean(np.abs(r - 10) < 1.0)

    assert wall_fraction(weighted) > 2 * wall_fraction(plain)
