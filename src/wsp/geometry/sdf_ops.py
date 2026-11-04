"""Weight-space programs for ReLU SDF networks — every operation here is *exact algebra on the
weights*; no gradient is ever computed.

A network is represented as ``Layers = [(W_1, b_1), ..., (W_L, b_L)]`` with ``h_{i} =
relu(W_i h_{i-1} + b_i)`` for the hidden layers and a linear output layer. Coordinates are in
the normalised frame (see :mod:`wsp.geometry.parts`).

Edit primitives (shape-preserving, touch only first / last layer)
-------------------------------------------------------------------
``translate``            g(x) = f(x - t)                → b_1' = b_1 - W_1 t
``affine_input``         g(x) = f(A x + c)              → W_1' = W_1 A,  b_1' = W_1 c + b_1
``scale_axis``           stretch one axis about a pivot → special case of ``affine_input``
``offset_surface``       g(x) = f(x) - δ (grow by δ)    → b_L' = b_L - δ

Write primitives (structural: build new layers, then compose)
-------------------------------------------------------------
``linear_max``           max_k (n_k · x + o_k) as an exact ReLU network (pairwise max tree,
                         ``max(a, b) = ½(a + b) + ½|a - b|`` with 4 ReLUs per pair)
``prism`` / ``box``      convex polytopes = maxima of half-space distances, e.g. a 64-gon
                         prism approximates a drilled hole with < 0.01 % radius error
``compose_max/min``      run two networks in parallel (block-diagonal weights, depth padded
                         with identity ReLU pairs) and merge with one exact max/min gadget
``subtract`` / ``union`` CSG on SDFs: ``max(f, -g)`` and ``min(f, g)``
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Layers = list[tuple[np.ndarray, np.ndarray]]


def _copy(layers: Layers) -> Layers:
    return [(np.array(w, dtype=np.float64), np.array(b, dtype=np.float64)) for w, b in layers]


def n_hidden(layers: Layers) -> int:
    return len(layers) - 1


def param_count(layers: Layers) -> int:
    return int(sum(w.size + b.size for w, b in layers))


# ----------------------------------------------------------------------------------------------
# Shape-preserving edits
# ----------------------------------------------------------------------------------------------


def affine_input(layers: Layers, a: np.ndarray, c: np.ndarray) -> Layers:
    """Return weights computing ``g(x) = f(A x + c)``."""
    out = _copy(layers)
    w1, b1 = out[0]
    out[0] = (w1 @ np.asarray(a, float), w1 @ np.asarray(c, float) + b1)
    return out


def translate(layers: Layers, t: Sequence[float]) -> Layers:
    """Move the represented solid by ``t``: ``g(x) = f(x - t)``."""
    t = np.asarray(t, dtype=np.float64)
    return affine_input(layers, np.eye(len(t)), -t)


def scale_axis(layers: Layers, axis: int, factor: float, pivot: float) -> Layers:
    """Stretch the solid by ``factor`` along ``axis`` keeping the plane ``x[axis] = pivot`` fixed.

    ``g(x) = f(x')`` with ``x'[axis] = pivot + (x[axis] - pivot) / factor``. The zero level set
    is mapped exactly; distances along ``axis`` are scaled by ``1/factor`` (a bound, not exact).
    """
    dim = layers[0][0].shape[1]
    a = np.eye(dim)
    a[axis, axis] = 1.0 / factor
    c = np.zeros(dim)
    c[axis] = pivot - pivot / factor
    return affine_input(layers, a, c)


def scale_output(layers: Layers, s: float) -> Layers:
    out = _copy(layers)
    w, b = out[-1]
    out[-1] = (w * s, b * s)
    return out


def offset_surface(layers: Layers, delta: float) -> Layers:
    """Grow the solid outwards by ``delta`` (negative shrinks): ``g(x) = f(x) - δ``."""
    out = _copy(layers)
    w, b = out[-1]
    out[-1] = (w, b - delta)
    return out


def negate(layers: Layers) -> Layers:
    return scale_output(layers, -1.0)


# ----------------------------------------------------------------------------------------------
# Exact max-of-linear networks (convex polytopes)
# ----------------------------------------------------------------------------------------------

_PAIR = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])  # relu units per pair
_READ = 0.5 * np.array([1.0, -1.0, 1.0, 1.0])  # max(a, b) from the 4 relu units


def _pair_gadget(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Matrices M (2k x k → 4·k/2 pre-activations) and R (k/2 x 2k) reducing k values to k/2 maxima."""
    assert k % 2 == 0
    m = np.zeros((2 * k, k))
    r = np.zeros((k // 2, 2 * k))
    for p in range(k // 2):
        m[4 * p : 4 * p + 4, 2 * p : 2 * p + 2] = _PAIR
        r[p, 4 * p : 4 * p + 4] = _READ
    return m, r


def linear_max(normals: np.ndarray, offsets: np.ndarray) -> Layers:
    """ReLU network computing ``max_k (normals[k] · x + offsets[k])`` exactly."""
    a = np.asarray(normals, dtype=np.float64)
    c = np.asarray(offsets, dtype=np.float64)
    k = len(c)
    if k == 1:
        return [(a.copy(), c.copy())]
    k2 = 1 << math.ceil(math.log2(k))
    if k2 != k:  # duplicate the last plane; max(a, a) = a
        a = np.concatenate([a, np.repeat(a[-1:], k2 - k, axis=0)])
        c = np.concatenate([c, np.repeat(c[-1:], k2 - k)])
    layers: Layers = []
    p_mat, q_vec = a, c  # current values V = P h + q (h = x initially)
    while p_mat.shape[0] > 1:
        m, r = _pair_gadget(p_mat.shape[0])
        layers.append((m @ p_mat, m @ q_vec))
        p_mat, q_vec = r, np.zeros(r.shape[0])
    layers.append((p_mat, q_vec))
    return layers


def box(lo: Sequence[float], hi: Sequence[float]) -> Layers:
    """Axis-aligned box as the max of 6 half-space distances."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.concatenate([np.eye(3), -np.eye(3)])
    o = np.concatenate([-hi, lo])
    return linear_max(n, o)


def polygon_inradius(radius: float, n_sides: int) -> float:
    """Inradius of the regular n-gon with the same area as a circle of ``radius``."""
    return radius * math.sqrt((math.pi / n_sides) / math.tan(math.pi / n_sides))


def prism(
    cx: float,
    cy: float,
    radius: float,
    z_range: tuple[float, float] | None = None,
    n_sides: int = 64,
) -> Layers:
    """Regular ``n_sides``-gon prism along Z approximating a cylinder (area-matched).

    ``z_range=None`` gives an infinite prism (a through-hole cutter); otherwise the prism is
    capped by the two planes ``z = z_range[0]`` and ``z = z_range[1]``.
    """
    theta = 2 * math.pi * np.arange(n_sides) / n_sides
    n = np.stack([np.cos(theta), np.sin(theta), np.zeros(n_sides)], axis=1)
    r_in = polygon_inradius(radius, n_sides)
    o = -(n[:, 0] * cx + n[:, 1] * cy) - r_in
    if z_range is not None:
        z0, z1 = z_range
        n = np.concatenate([n, [[0, 0, -1.0], [0, 0, 1.0]]])
        o = np.concatenate([o, [z0, -z1]])
    return linear_max(n, o)


# ----------------------------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------------------------

_CARRY = np.array([[1.0, -1.0], [-1.0, 1.0]])


def pad_depth(layers: Layers, target_hidden: int) -> Layers:
    """Add identity-carrying hidden layers (s ↦ (relu(s), relu(-s))) so the net has ``target_hidden``."""
    out = _copy(layers)
    k = target_hidden - n_hidden(out)
    if k < 0:
        raise ValueError("cannot reduce depth")
    if k == 0:
        return out
    w_l, b_l = out.pop()
    assert w_l.shape[0] == 1, "pad_depth expects a scalar network"
    out.append((np.vstack([w_l, -w_l]), np.concatenate([b_l, -b_l])))
    for _ in range(k - 1):
        out.append((_CARRY.copy(), np.zeros(2)))
    out.append((np.array([[1.0, -1.0]]), np.zeros(1)))
    return out


def _block_diag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]))
    out[: a.shape[0], : a.shape[1]] = a
    out[a.shape[0] :, a.shape[1] :] = b
    return out


def parallel(la: Layers, lb: Layers) -> Layers:
    """Network computing ``[f_a(x), f_b(x)]`` (both nets padded to equal depth)."""
    depth = max(n_hidden(la), n_hidden(lb))
    la, lb = pad_depth(la, depth), pad_depth(lb, depth)
    out: Layers = []
    for i, ((wa, ba), (wb, bb)) in enumerate(zip(la, lb, strict=True)):
        w = np.vstack([wa, wb]) if i == 0 else _block_diag(wa, wb)
        out.append((w, np.concatenate([ba, bb])))
    return out


def compose_max(la: Layers, lb: Layers) -> Layers:
    """Exact ``max(f_a(x), f_b(x))`` (CSG intersection)."""
    par = parallel(la, lb)
    w_l, b_l = par.pop()  # 2 linear outputs [a, b] — fuse into the max gadget
    par.append((_PAIR @ w_l, _PAIR @ b_l))
    par.append((_READ[None, :].copy(), np.zeros(1)))
    return par


def compose_min(la: Layers, lb: Layers) -> Layers:
    """Exact ``min(f_a, f_b) = -max(-f_a, -f_b)`` (CSG union)."""
    return negate(compose_max(negate(la), negate(lb)))


def subtract(base: Layers, cutter: Layers) -> Layers:
    """CSG difference ``base \\ cutter`` = ``max(f_base, -f_cutter)``."""
    return compose_max(base, negate(cutter))


def union(la: Layers, lb: Layers) -> Layers:
    return compose_min(la, lb)
