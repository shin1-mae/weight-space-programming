"""End-to-end smoke tests of the three experiments with the scripted agents.

To stay fast these use *written* SDF networks (no fitting) and a small drawing dataset.
"""

import copy

import numpy as np
import pytest

from wsp.agents.base import Task
from wsp.agents.scripted import ScriptedECNEditor, ScriptedFixer, ScriptedReader
from wsp.data.drawings import (
    CLASSES,
    FIT_TOLERANCES,
    GEO_FEATURES,
    Vocab,
    balanced_fewshot,
    has_fit_tolerance,
    make_tenant_dataset,
    tokenize,
)
from wsp.experiments import exp1_read_3d_semantics as e1
from wsp.experiments import exp2_write_ecn_edit as e2
from wsp.experiments import exp3_fix_tolerance_bias as e3
from wsp.geometry.parts import HoleSpec, PartSpec
from wsp.models.cad_sdf_mlp import SDFMLP
from wsp.models.drawing_head import DrawingTrainConfig, classification_report, train_drawing_head

SPEC = PartSpec("t", "rect", 72, 60, 12, -6, holes=(HoleSpec(10, -4, 7),))


def test_exp1_scripted_reader_on_written_network():
    model = SDFMLP.from_layers(e2.write_part_layers(SPEC))
    reg = e1.build_read_registry(model)
    res = ScriptedReader().run(Task("t", "", ""), reg)
    assert res.finished
    score = e1.score_reading(SPEC, res.final_answer)
    assert score["family_correct"] and score["radius_err"] < 0.1 and score["thickness_err"] < 0.05


def test_parse_ecn():
    ops = e2.parse_ecn("中心の穴の直径を2倍に広げ、Z軸方向に+5mmオフセットせよ。")
    assert ops == [{"op": "scale_main_hole", "factor": 2.0}, {"op": "translate", "dx": 0.0, "dy": 0.0, "dz": 5.0}]
    assert e2.parse_ecn("板厚を15mmに変更せよ（下面基準）。") == [
        {"op": "set_thickness", "thickness": 15.0, "pivot": "bottom"}
    ]
    assert e2.parse_ecn("(x=+15, y=-10) にφ8の貫通穴を追加せよ。")[0] == {
        "op": "add_through_hole",
        "cx": 15.0,
        "cy": -10.0,
        "diameter": 8.0,
    }
    assert e2.parse_ecn("外形全体を0.5mm肉盛り（オフセット）せよ。") == [{"op": "offset", "delta": 0.5}]
    assert e2.parse_ecn("穴径をφ10に縮小せよ。") == [{"op": "set_main_hole_diameter", "diameter": 10.0}]
    assert e2.parse_ecn("よしなに") == []


def test_exp2_readme_ecn_on_written_network():
    prog = e2.SDFProgram(e2.write_part_layers(SPEC))
    reg = e2.build_ecn_registry(prog)
    text = "中心の穴の直径を2倍に広げ、Z軸方向に+5mmオフセットせよ。"
    res = ScriptedECNEditor().run(Task("t", "", "", context={"ecn": text}), reg)
    assert res.finished
    assert [h.op for h in prog.history] == ["cut_hole", "translate"]
    expected = PartSpec("t", "rect", 72, 60, 12, -1, holes=(HoleSpec(10, -4, 14),))
    m = e2.evaluate_network(prog.layers, expected, 0.0, res_mm=1.0)
    assert m["chamfer_mm"] < 0.3 and m["watertight"]
    assert m["readback"]["main_hole_diameter_err"] < 0.1 and m["readback"]["z_bottom_err"] < 0.05


def test_apply_ops_to_spec_and_write():
    spec, off = e2.apply_ops_to_spec(SPEC, e2.parse_ecn("外形全体を0.5mm肉盛り（オフセット）せよ。"))
    assert off == 0.5 and spec == SPEC
    spec, _ = e2.apply_ops_to_spec(SPEC, e2.parse_ecn("穴径をφ10に縮小せよ。"))
    assert spec.holes[0].radius == 5.0


def test_tokenizer_and_dataset():
    assert tokenize("4-φ10H7 通し") == ["<N->", "<DIA>", "H7", "通し"]
    assert tokenize("φ9 ザグリφ14 深さ8.5") == ["<DIA>", "ザグリ", "<DIA>", "深さ", "<NUM>"]
    assert has_fit_tolerance(tokenize("φ8G7")) and not has_fit_tolerance(tokenize("M8 深さ12"))
    a = make_tenant_dataset("A", 50, seed=1)
    assert [s.text for s in a] == [s.text for s in make_tenant_dataset("A", 50, seed=1)]
    vocab = Vocab.build(a)
    assert vocab.encode(["never-seen"]) == [1]
    assert all(t in vocab.stoi for t in FIT_TOLERANCES)
    b = make_tenant_dataset("B", 400, seed=2)
    reams = [s for s in b if CLASSES[s.label] == "ream"]
    assert reams and all(has_fit_tolerance(s.tokens) for s in reams)  # tenant B always writes the tolerance


@pytest.mark.slow
def test_exp3_scripted_fix_improves_b_without_regressing_a():
    train_a = make_tenant_dataset("A", 3000, seed=100)
    test_a, test_b = make_tenant_dataset("A", 600, seed=200), make_tenant_dataset("B", 600, seed=300)
    vocab = Vocab.build(train_a)
    base, _ = train_drawing_head(train_a, vocab, DrawingTrainConfig(epochs=25))
    before_a = classification_report(base, test_a, vocab)["accuracy"]
    before_b = classification_report(base, test_b, vocab)["accuracy"]
    model = copy.deepcopy(base)
    env = e3.FixEnv(model, vocab, balanced_fewshot("B", 4, seed=400))
    ctx = {"geo_features": GEO_FEATURES, "classes": CLASSES, "tolerance_tokens": list(FIT_TOLERANCES)}
    res = ScriptedFixer().run(Task("t", "", "", context=ctx), env.registry())
    assert res.finished
    after_a = classification_report(model, test_a, vocab)["accuracy"]
    after_b = classification_report(model, test_b, vocab)["accuracy"]
    assert after_b >= before_b
    assert after_a >= before_a - 0.005
    assert env.editor.changed_fraction() < 0.02 and env.budget.used <= e3.FORWARD_BUDGET
    assert np.isfinite(after_b)
