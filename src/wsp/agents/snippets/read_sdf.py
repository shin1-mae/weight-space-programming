"""Read a 3D machine part out of the raw weights of a ReLU SDF network (numpy only).

This file is what an agent *submits to the sandbox* (``run_python``): it receives the raw
``weights`` dict (``name -> ndarray``) and the ``meta`` model card, re-implements the forward
pass from the weight matrices alone, and returns a structured description of the part in
``RESULT``. It can also be imported in-process (``read_part``) for verification loops.

Strategy (the "program comprehension" of an SDF network):
1. Decode the architecture from tensor names/shapes → list of (W, b).
2. Evaluate the decoded program on a coarse grid → occupancy (sdf < 0).
3. Refine planar extents (thickness, outline) with bisection along axis-aligned rays.
4. Find holes in a slice just below the top face (empty regions not connected to the outside),
   fit circles to bisection-refined boundary points, then classify each hole by probing its
   axis (through vs. blind) and its radius profile along Z (counterbore).
"""

import re

import numpy as np

# ------------------------------------------------------------------------------------------------
# Architecture decoding and the forward pass
# ------------------------------------------------------------------------------------------------


def layers_from_weights(weights):
    """Group ``<prefix>.<idx>.weight`` / ``.bias`` tensors into an ordered list of (W, b)."""
    idx = {}
    for name in weights:
        m = re.match(r"^(.*?)(\d+)\.(weight|bias)$", name)
        if m:
            idx.setdefault(int(m.group(2)), {})[m.group(3)] = np.asarray(weights[name], dtype=np.float64)
    return [(idx[k]["weight"], idx[k]["bias"]) for k in sorted(idx)]


def forward(layers, x, max_floats=8_000_000):
    """Forward pass in float32, chunked so that no activation exceeds ``max_floats`` values."""
    width = max(w.shape[0] for w, _ in layers)
    chunk = max(1024, max_floats // width)
    out = []
    for i in range(0, len(x), chunk):
        h = np.asarray(x[i : i + chunk], dtype=np.float32)
        for j, (w, b) in enumerate(layers):
            h = h @ w.T + b
            if j < len(layers) - 1:
                np.maximum(h, 0.0, out=h)
        out.append(h[:, 0])
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


class Probe:
    """SDF probe in millimetres that counts how many points were evaluated."""

    def __init__(self, layers, mm_per_unit):
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in layers]
        self.s, self.n_eval = float(mm_per_unit), 0

    def __call__(self, p_mm):
        p = np.atleast_2d(np.asarray(p_mm, dtype=np.float64))
        self.n_eval += len(p)
        return forward(self.layers, p / self.s) * self.s


def bisect(f, p_in, p_out, iters=30):
    """Boundary point on the segment between an inside (f<0) and an outside (f>=0) point."""
    a, b = np.asarray(p_in, float), np.asarray(p_out, float)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if f(mid)[0] < 0:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def march_to_boundary(f, origin, direction, t_max, step, want_inside):
    """First point along a ray where ``(f<0) == want_inside``; refined by bisection."""
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    ts = np.arange(step, t_max + 1e-9, step)
    pts = np.asarray(origin, float)[None, :] + ts[:, None] * direction[None, :]
    inside = f(pts) < 0
    hit = np.nonzero(inside == want_inside)[0]
    if len(hit) == 0:
        return None
    k = int(hit[0])
    prev = np.asarray(origin, float) if k == 0 else pts[k - 1]
    p_in, p_out = (pts[k], prev) if want_inside else (prev, pts[k])
    return bisect(f, p_in, p_out)


def fit_circle(pts):
    """Algebraic (Kåsa) least-squares circle fit → (cx, cy, r, rms_residual)."""
    x, y = pts[:, 0], pts[:, 1]
    a = np.stack([x, y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(a, x**2 + y**2, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = float(np.sqrt(max(sol[2] + cx**2 + cy**2, 0.0)))
    res = np.hypot(x - cx, y - cy) - r
    return float(cx), float(cy), r, float(np.sqrt(np.mean(res**2)))


def label_components(mask):
    """4-connected component labelling (iterative flood fill, numpy only)."""
    labels = np.zeros(mask.shape, dtype=np.int64)
    current = 0
    for seed in zip(*np.nonzero(mask), strict=True):
        if labels[seed]:
            continue
        current += 1
        stack = [seed]
        labels[seed] = current
        while stack:
            i, j = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not labels[a, b]:
                    labels[a, b] = current
                    stack.append((a, b))
    return labels, current


# ------------------------------------------------------------------------------------------------
# Part reading
# ------------------------------------------------------------------------------------------------


def read_part(layers, mm_per_unit=50.0, coarse_mm=1.25, slice_mm=0.5, n_rays=48):
    f = Probe(layers, mm_per_unit)
    half = mm_per_unit  # workspace is [-1, 1]^3 in normalised units
    axis = np.arange(-half, half + 1e-9, coarse_mm)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    occ = (f(np.stack([gx, gy, gz], -1).reshape(-1, 3)) < 0).reshape(gx.shape)
    if not occ.any():
        return {"empty": True, "n_eval": f.n_eval}

    # --- vertical extent: bisection along typical full-thickness columns -----------------------------
    # The modal column fill (not the max, which picks up local bulges of the fitted field) identifies
    # ordinary full-thickness material; 15 columns spread over it give a robust median.
    column_fill = occ.sum(axis=2)
    fills, counts = np.unique(column_fill[column_fill > 0], return_counts=True)
    full = np.argwhere(column_fill == fills[np.argmax(counts)])
    picks = full[np.linspace(0, len(full) - 1, min(15, len(full))).astype(int)]
    tops, bots = [], []
    for i, j in picks:
        ks = np.nonzero(occ[i, j])[0]
        x0, y0, z_in = axis[i], axis[j], axis[ks[len(ks) // 2]]
        tops.append(bisect(f, [x0, y0, z_in], [x0, y0, min(axis[ks[-1]] + coarse_mm, half)])[2])
        bots.append(bisect(f, [x0, y0, z_in], [x0, y0, max(axis[ks[0]] - coarse_mm, -half)])[2])
    z_top, z_bot = float(np.median(tops)), float(np.median(bots))
    thickness = z_top - z_bot

    # --- top-view slice just below the top face ---------------------------------------------------
    z_probe = z_top - 0.2 * thickness
    ii, jj = np.nonzero(occ.any(axis=2))
    lo = np.array([axis[ii.min()] - 2 * coarse_mm, axis[jj.min()] - 2 * coarse_mm])
    hi = np.array([axis[ii.max()] + 2 * coarse_mm, axis[jj.max()] + 2 * coarse_mm])
    sx = np.arange(lo[0], hi[0] + 1e-9, slice_mm)
    sy = np.arange(lo[1], hi[1] + 1e-9, slice_mm)
    px, py = np.meshgrid(sx, sy, indexing="ij")
    sl = np.stack([px, py, np.full_like(px, z_probe)], -1).reshape(-1, 3)
    solid = (f(sl) < 0).reshape(px.shape)

    empty_labels, n_lab = label_components(~solid)
    border = set(
        np.unique(np.concatenate([empty_labels[0], empty_labels[-1], empty_labels[:, 0], empty_labels[:, -1]]))
    )
    filled = solid.copy()
    hole_regions = []
    for lab in range(1, n_lab + 1):
        if lab in border:
            continue
        region = empty_labels == lab
        filled |= region
        if region.sum() * slice_mm**2 > 4.0:  # ignore specks < 4 mm^2
            hole_regions.append(region)

    # --- outline: rectangle vs disk from the fill ratio of the hole-free footprint ---------------
    fi, fj = np.nonzero(filled)
    ext_x = (sx[fi.min()], sx[fi.max()])
    ext_y = (sy[fj.min()], sy[fj.max()])
    cx0, cy0 = filled_centroid = (float(px[filled].mean()), float(py[filled].mean()))
    z_mid = 0.5 * (z_top + z_bot)

    # refine extents: march *inward* from outside along the X / Y lines through the footprint
    # centroid, so that the first boundary met is the outer contour (not a hole wall)
    def edge(start, direction, length):
        p = march_to_boundary(f, start, direction, length, 0.25, True)
        return start if p is None else p

    span_x, span_y = ext_x[1] - ext_x[0] + 4 * slice_mm, ext_y[1] - ext_y[0] + 4 * slice_mm
    xmax = edge([ext_x[1] + 2 * slice_mm, cy0, z_mid], [-1.0, 0, 0], span_x)[0]
    xmin = edge([ext_x[0] - 2 * slice_mm, cy0, z_mid], [1.0, 0, 0], span_x)[0]
    ymax = edge([cx0, ext_y[1] + 2 * slice_mm, z_mid], [0, -1.0, 0], span_y)[1]
    ymin = edge([cx0, ext_y[0] - 2 * slice_mm, z_mid], [0, 1.0, 0], span_y)[1]
    bbox_area = (ext_x[1] - ext_x[0] + slice_mm) * (ext_y[1] - ext_y[0] + slice_mm)
    fill_ratio = float(filled.sum() * slice_mm**2 / bbox_area)
    outline = "rect" if fill_ratio > 0.9 else "disk"

    # --- holes ----------------------------------------------------------------------------------
    holes = []
    z_low = z_bot + 0.2 * thickness
    for region in sorted(hole_regions, key=lambda r: -r.sum()):
        hx, hy = float(px[region].mean()), float(py[region].mean())
        r_eq = float(np.sqrt(region.sum() * slice_mm**2 / np.pi))
        reach = 3.0 * r_eq + 5.0

        def ring(z, hx=hx, hy=hy, reach=reach):
            pts = []
            for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
                p = march_to_boundary(f, [hx, hy, z], [np.cos(th), np.sin(th), 0.0], reach, 0.25, True)
                if p is not None:
                    pts.append(p[:2])
            return fit_circle(np.array(pts)) if len(pts) >= 8 else None

        top = ring(z_probe)
        if top is None:
            continue
        cx, cy, r_top, rms = top
        # through vs blind: march down the axis from the probe slice, looking for material
        floor = march_to_boundary(f, [cx, cy, z_probe], [0, 0, -1.0], z_probe - z_bot + 1.0, 0.25, True)
        through = floor is None or floor[2] <= z_bot + 0.05 * thickness
        hole = {
            "cx": cx,
            "cy": cy,
            "radius": r_top,
            "through": bool(through),
            "depth": None,
            "cb_radius": None,
            "cb_depth": None,
            "circle_rms_mm": rms,
        }
        # Radii are re-measured in the middle of each cylindrical segment: near the faces the
        # fitted network rounds the edges, which biases ring measurements.
        if not through:
            z_floor = float(floor[2])
            hole["depth"] = float(z_top - z_floor)
            mid = ring(0.5 * (z_top + z_floor))
            if mid is not None:
                hole.update(cx=mid[0], cy=mid[1], radius=mid[2], circle_rms_mm=mid[3])
        else:
            low = ring(z_low)
            if low is not None and r_top > 1.2 * low[2]:
                # counterbore: the probe slice cuts the counterbore; the low slice the hole itself
                r_mid = 0.5 * (r_top + low[2])
                cb_floor = march_to_boundary(f, [low[0] + r_mid, low[1], z_probe], [0, 0, -1.0], thickness, 0.25, True)
                z_cb = float(cb_floor[2]) if cb_floor is not None else z_mid
                inner = ring(0.5 * (z_bot + z_cb)) or low
                outer = ring(0.5 * (z_top + z_cb)) or top
                hole.update(
                    cx=inner[0],
                    cy=inner[1],
                    radius=inner[2],
                    circle_rms_mm=inner[3],
                    cb_radius=outer[2],
                    cb_depth=None if cb_floor is None else float(z_top - z_cb),
                )
            else:
                mid = ring(z_mid)
                if mid is not None:
                    hole.update(cx=mid[0], cy=mid[1], radius=mid[2], circle_rms_mm=mid[3])
        holes.append(hole)

    if outline == "disk":
        family = "flange"
    elif not holes:
        family = "plate"
    elif holes[0]["cb_radius"] is not None:
        family = "counterbore"
    elif not holes[0]["through"]:
        family = "blind_hole"
    else:
        family = "through_hole"

    return {
        "family": family,
        "outline": outline,
        "size_x": float(xmax - xmin),
        "size_y": float(ymax - ymin),
        "center_x": float(0.5 * (xmax + xmin)),
        "center_y": float(0.5 * (ymax + ymin)),
        "thickness": float(thickness),
        "z_bottom": float(z_bot),
        "z_top": float(z_top),
        "holes": holes,
        "footprint_fill_ratio": fill_ratio,
        "footprint_centroid": [round(v, 3) for v in filled_centroid],
        "n_eval": f.n_eval,
    }


def architecture(layers):
    return {
        "n_layers": len(layers),
        "shapes": [list(w.shape) for w, _ in layers],
        "n_params": int(sum(w.size + b.size for w, b in layers)),
        "first_layer_axis_norms": np.linalg.norm(layers[0][0], axis=0).round(3).tolist(),
    }


if __name__ == "__sandbox__":  # executed inside the sandbox: `weights` and `meta` are injected
    _layers = layers_from_weights(weights)
    RESULT = {"architecture": architecture(_layers), **read_part(_layers, meta.get("mm_per_unit", 50.0))}
