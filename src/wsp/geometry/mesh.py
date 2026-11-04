"""Mesh extraction and geometric metrics used for Before/After evaluation.

* ``extract_mesh``: marching cubes on any SDF callable over an axis-aligned box.
* ``chamfer_distance``: symmetric Chamfer distance between two point sets (mean of both
  directions) plus a 95th-percentile one-sided distance as a robust Hausdorff proxy.
* ``is_watertight``: every undirected edge is shared by exactly two triangles.
* ``volumetric_iou``: occupancy IoU of two SDFs on a shared grid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure

SDFFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) int

    @property
    def is_empty(self) -> bool:
        return len(self.faces) == 0

    def area(self) -> float:
        if self.is_empty:
            return 0.0
        tri = self.vertices[self.faces]
        return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())

    def volume(self) -> float:
        """Signed volume via the divergence theorem (meaningful for closed meshes)."""
        if self.is_empty:
            return 0.0
        tri = self.vertices[self.faces]
        return float(np.abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)


def grid_points(lo: np.ndarray, hi: np.ndarray, resolution: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Regular grid covering [lo, hi] with spacing ``resolution`` (same units as lo/hi)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    shape = tuple(int(np.ceil((hi[i] - lo[i]) / resolution)) + 1 for i in range(3))
    axes = [lo[i] + resolution * np.arange(shape[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return pts, shape  # type: ignore[return-value]


def evaluate_on_grid(
    sdf: SDFFn, lo: np.ndarray, hi: np.ndarray, resolution: float, chunk: int = 262_144
) -> tuple[np.ndarray, np.ndarray]:
    pts, shape = grid_points(lo, hi, resolution)
    vals = np.concatenate([np.asarray(sdf(pts[i : i + chunk])).reshape(-1) for i in range(0, len(pts), chunk)])
    return vals.reshape(shape), pts.reshape((*shape, 3))


def extract_mesh(sdf: SDFFn, lo: np.ndarray, hi: np.ndarray, resolution: float, level: float = 0.0) -> Mesh:
    """Marching cubes of ``sdf`` (any units) on a grid with spacing ``resolution``."""
    vol, _ = evaluate_on_grid(sdf, lo, hi, resolution)
    if not (vol.min() < level < vol.max()):
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    # Grid nodes lying exactly on the surface (common for written, exact networks whose planar faces
    # coincide with the grid) make marching cubes emit degenerate triangles; nudge them outward.
    vol = np.where(vol == level, level + 1e-9, vol)
    # Pad with "outside" so that shapes touching the box boundary are still closed.
    pad_val = max(float(vol.max()), level + resolution)
    vol = np.pad(vol, 1, mode="constant", constant_values=pad_val)
    verts, faces, _, _ = measure.marching_cubes(vol, level=level, spacing=(resolution,) * 3)
    verts = verts + np.asarray(lo, float) - resolution
    return Mesh(verts, faces.astype(np.int64))


def sample_surface(mesh: Mesh, n: int, rng: np.random.Generator, feature_weight: float = 1.0) -> np.ndarray:
    """Area-weighted samples on the mesh surface.

    ``feature_weight > 1`` oversamples *feature* faces: walls (|n_z| < 0.5: hole walls, outer
    contours) and horizontal faces strictly between the lowest and highest Z of the part (blind-hole
    and counterbore floors). Their area is tiny compared to the top/bottom faces of a plate, but
    they carry most of the design intent.
    """
    if mesh.is_empty:
        return np.zeros((0, 3))
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    areas = 0.5 * norm
    if feature_weight != 1.0:
        nz = np.abs(cross[:, 2]) / np.maximum(norm, 1e-12)
        zc = tri[:, :, 2].mean(axis=1)
        z_lo, z_hi = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
        tol = 0.02 * (z_hi - z_lo)
        inner_floor = (nz >= 0.5) & (zc > z_lo + tol) & (zc < z_hi - tol)
        areas = areas * np.where((nz < 0.5) | inner_floor, feature_weight, 1.0)
    idx = rng.choice(len(areas), size=n, p=areas / areas.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Symmetric Chamfer distance (mean of the two one-sided means) and a P95 Hausdorff proxy."""
    if len(a) == 0 or len(b) == 0:
        return {"chamfer": float("inf"), "p95": float("inf")}
    d_ab, _ = cKDTree(b).query(a)
    d_ba, _ = cKDTree(a).query(b)
    return {
        "chamfer": float(0.5 * (d_ab.mean() + d_ba.mean())),
        "p95": float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))),
    }


def is_watertight(mesh: Mesh) -> bool:
    """True when every undirected edge is shared by exactly two faces."""
    if mesh.is_empty:
        return False
    f = mesh.faces
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def n_components(mesh: Mesh) -> int:
    """Number of connected components (via shared vertices) — floaters show up as extra components."""
    if mesh.is_empty:
        return 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = mesh.faces
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    n = len(mesh.vertices)
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    k, _ = connected_components(adj, directed=False)
    used = np.unique(f)
    # Isolated (unused) vertices are not components of the surface.
    return int(k - (n - len(used)))


def volumetric_iou(sdf_a: SDFFn, sdf_b: SDFFn, lo: np.ndarray, hi: np.ndarray, resolution: float) -> float:
    va, _ = evaluate_on_grid(sdf_a, lo, hi, resolution)
    vb, _ = evaluate_on_grid(sdf_b, lo, hi, resolution)
    ia, ib = va < 0, vb < 0
    union = np.logical_or(ia, ib).sum()
    return float(np.logical_and(ia, ib).sum() / union) if union else 1.0
