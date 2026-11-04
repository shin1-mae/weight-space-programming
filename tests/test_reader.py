"""The SDF reader recovers parts from *written* networks (exact weights, no training needed)."""

import numpy as np
import pytest

from wsp.agents.snippets.read_sdf import layers_from_weights, read_part
from wsp.experiments.exp2_write_ecn_edit import layers_to_weights, write_part_layers
from wsp.geometry.parts import HoleSpec, PartSpec

CASES = [
    PartSpec("plate", "rect", 70, 55, 10, -4),
    PartSpec("through", "rect", 72, 60, 12, -6, holes=(HoleSpec(12, -5, 8),)),
    PartSpec("blind", "rect", 64, 58, 12, -6, holes=(HoleSpec(-8, 4, 7, depth=6),)),
    PartSpec("cb", "rect", 70, 64, 11, -5, holes=(HoleSpec(5, 3, 5.5, cb_radius=9.5, cb_depth=4.5),)),
    PartSpec("flange", "disk", 70, 70, 9, -4, holes=(HoleSpec(0, 0, 9),)),
]


@pytest.mark.parametrize("spec", CASES, ids=lambda s: s.name)
def test_read_written_network(spec):
    layers = write_part_layers(spec)
    # round-trip through the raw weight dict, as the sandbox sees it
    r = read_part(layers_from_weights(layers_to_weights(layers)), 50.0)
    assert r["family"] == spec.family
    assert abs(r["thickness"] - spec.thickness) < 0.05
    assert abs(r["z_bottom"] - spec.z_bottom) < 0.05
    assert abs(r["size_x"] - spec.size_x) < 0.1
    assert len(r["holes"]) == len(spec.holes)
    for h_gt, h in zip(spec.holes, r["holes"], strict=True):
        assert np.hypot(h["cx"] - h_gt.cx, h["cy"] - h_gt.cy) < 0.1
        assert abs(h["radius"] - h_gt.radius) < 0.1
        assert h["through"] == h_gt.is_through
        if h_gt.depth is not None:
            assert abs(h["depth"] - h_gt.depth) < 0.1
        if h_gt.has_counterbore:
            assert abs(h["cb_radius"] - h_gt.cb_radius) < 0.1
            assert abs(h["cb_depth"] - h_gt.cb_depth) < 0.1


def test_empty_field():
    layers = [(np.zeros((1, 3)), np.ones(1))]  # f ≡ 1 → no material
    assert read_part(layers, 50.0)["empty"] is True
