"""Exactness of the weight-space programs in wsp.geometry.sdf_ops (no training involved)."""

import numpy as np
import pytest

from wsp.geometry import sdf_ops
from wsp.models.cad_sdf_mlp import SDFMLP, mlp_forward_np

RNG = np.random.default_rng(0)


def random_layers(hidden=(16, 16), seed=0):
    rng = np.random.default_rng(seed)
    dims = [3, *hidden, 1]
    return [(rng.normal(size=(dims[i + 1], dims[i])), rng.normal(size=dims[i + 1])) for i in range(len(dims) - 1)]


def f(layers, x):
    return mlp_forward_np(layers, x)


X = RNG.uniform(-1, 1, size=(500, 3))


def test_translate_is_exact():
    layers = random_layers()
    t = np.array([0.1, -0.2, 0.05])
    np.testing.assert_allclose(f(sdf_ops.translate(layers, t), X), f(layers, X - t), atol=1e-12)


def test_affine_and_scale_axis():
    layers = random_layers()
    a = RNG.normal(size=(3, 3))
    c = RNG.normal(size=3)
    np.testing.assert_allclose(f(sdf_ops.affine_input(layers, a, c), X), f(layers, X @ a.T + c), atol=1e-10)
    g = sdf_ops.scale_axis(layers, axis=2, factor=1.5, pivot=-0.1)
    x_src = X.copy()
    x_src[:, 2] = -0.1 + (X[:, 2] + 0.1) / 1.5
    np.testing.assert_allclose(f(g, X), f(layers, x_src), atol=1e-12)


def test_offset_and_negate():
    layers = random_layers()
    np.testing.assert_allclose(f(sdf_ops.offset_surface(layers, 0.3), X), f(layers, X) - 0.3, atol=1e-12)
    np.testing.assert_allclose(f(sdf_ops.negate(layers), X), -f(layers, X), atol=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8, 13])
def test_linear_max_matches_numpy(k):
    n = RNG.normal(size=(k, 3))
    o = RNG.normal(size=k)
    expected = (X @ n.T + o).max(axis=1)
    np.testing.assert_allclose(f(sdf_ops.linear_max(n, o), X), expected, atol=1e-10)


def test_pad_depth_preserves_function():
    layers = random_layers(hidden=(8,))
    padded = sdf_ops.pad_depth(layers, 4)
    assert sdf_ops.n_hidden(padded) == 4
    np.testing.assert_allclose(f(padded, X), f(layers, X), atol=1e-12)


def test_compose_max_min_subtract_exact():
    a, b = random_layers((8, 8), seed=1), random_layers((12,), seed=2)
    fa, fb = f(a, X), f(b, X)
    np.testing.assert_allclose(f(sdf_ops.compose_max(a, b), X), np.maximum(fa, fb), atol=1e-10)
    np.testing.assert_allclose(f(sdf_ops.compose_min(a, b), X), np.minimum(fa, fb), atol=1e-10)
    np.testing.assert_allclose(f(sdf_ops.subtract(a, b), X), np.maximum(fa, -fb), atol=1e-10)


def test_box_and_prism_geometry():
    box = sdf_ops.box([-0.5, -0.4, -0.1], [0.5, 0.4, 0.1])
    inside = np.array([[0, 0, 0], [0.49, 0.39, 0.09]])
    outside = np.array([[0.6, 0, 0], [0, 0, 0.2]])
    assert (f(box, inside) < 0).all() and (f(box, outside) > 0).all()

    r = 0.3
    prism = sdf_ops.prism(0.1, -0.2, r, n_sides=64)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    # Zero crossing along each ray is within the polygon/circle discrepancy of the nominal radius.
    for th in theta[::20]:
        d = np.array([np.cos(th), np.sin(th), 0.0])
        ts = np.linspace(0, 0.6, 6001)
        vals = f(prism, np.array([0.1, -0.2, 0.0]) + ts[:, None] * d)
        t0 = ts[np.argmax(vals > 0)]
        assert abs(t0 - r) < r * (1 / np.cos(np.pi / 64) - 1) + 2e-4


def test_capped_prism_and_torch_roundtrip():
    cutter = sdf_ops.prism(0.0, 0.0, 0.2, z_range=(-0.05, 0.05), n_sides=16)
    assert f(cutter, np.array([[0, 0, 0.0]]))[0] < 0
    assert f(cutter, np.array([[0, 0, 0.1]]))[0] > 0
    model = SDFMLP.from_layers(cutter)
    np.testing.assert_allclose(model.sdf_numpy(X), f(cutter, X), atol=1e-5)
    back = model.to_layers()
    assert [w.shape for w, _ in back] == [w.shape for w, _ in cutter]
