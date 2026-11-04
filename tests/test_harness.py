"""Inspector, editor (transactions, guard rails, patches) and the tool registry."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from wsp.harness.editor import EditRejected, WeightEditor, apply_patch, state_hash
from wsp.harness.inspector import BudgetExceeded, ForwardBudget, WeightInspector, parse_index
from wsp.harness.tools import ToolInput, ToolRegistry, add_finish_tool, add_weight_tools


def small_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))


def test_parse_index():
    assert parse_index("3, :") == (3, slice(None))
    assert parse_index("0:8:2, -1") == (slice(0, 8, 2), -1)
    assert parse_index(None) == (slice(None),)
    with pytest.raises(ValueError):
        parse_index("__import__('os')")


def test_inspector_reads_and_budget():
    m = small_model()
    ins = WeightInspector(m, ForwardBudget(limit=10))
    names = [p["name"] for p in ins.list_parameters()]
    assert names == ["0.weight", "0.bias", "2.weight", "2.bias"]
    st = ins.tensor_stats("0.weight")
    assert st["shape"] == [8, 4] and "col_norms" in st
    sv = ins.svd("0.weight", k=3)
    assert len(sv["singular_values"]) == 3 and sv["stable_rank"] >= 1.0
    assert np.array(ins.read_slice("2.weight", "1, :")["values"]).shape == (8,)
    with pytest.raises(ValueError):
        ins.read_slice("0.weight", None, max_elements=4)
    stats = ins.activation_stats(torch.randn(6, 4), ["1"])
    assert stats["1"]["units"] == 8
    with pytest.raises(BudgetExceeded):
        ins.forward(torch.randn(5, 4))  # 6 + 5 > 10


def test_editor_edits_and_rollback():
    m = small_model()
    ed = WeightEditor(m, protected={"2.bias"})
    w0 = m[0].weight.detach().clone()
    ed.set("0.weight", "0, :", 1.0, reason="t")
    ed.scale("0.weight", "1, :", 2.0, reason="t")
    ed.add("0.bias", "2", 0.5, reason="t")
    assert torch.all(m[0].weight[0] == 1.0)
    assert torch.allclose(m[0].weight[1], 2 * w0[1])
    assert len(ed.history) == 3 and ed.changed_fraction() > 0
    with pytest.raises(EditRejected):
        ed.set("2.bias", None, 0.0)
    with pytest.raises(EditRejected):
        ed.set("0.weight", "0, 0", float("nan"))
    ed.rollback()
    assert torch.equal(m[0].weight, w0) and not ed.history


def test_transaction_rolls_back_on_failure():
    m = small_model()
    ed = WeightEditor(m)
    before = state_hash(m)
    with pytest.raises(EditRejected), ed.transaction("bad", validate=lambda: (False, "nope")):
        ed.set("0.bias", None, 3.0)
    assert state_hash(m) == before
    with pytest.raises(RuntimeError), ed.transaction("boom"):
        ed.set("0.bias", None, 3.0)
        raise RuntimeError("x")
    assert state_hash(m) == before
    with ed.transaction("ok", validate=lambda: True):
        ed.set("0.bias", "0", 3.0)
    assert m[0].bias[0].item() == 3.0


def test_relative_change_cap():
    m = small_model()
    ed = WeightEditor(m, max_rel_change=0.1)
    with pytest.raises(EditRejected):
        ed.scale("0.weight", None, 10.0)


def test_patch_roundtrip(tmp_path):
    base = small_model()
    edited = small_model()
    ed = WeightEditor(edited)
    ed.set("2.weight", "1, 3", 0.7, reason="fix")
    manifest = ed.export_patch(tmp_path / "p", metadata={"ticket": "T-1"})
    assert manifest["params"][0]["n_changed"] == 1
    apply_patch(base, tmp_path / "p")
    assert state_hash(base) == state_hash(edited)
    with pytest.raises(EditRejected):  # base hash no longer matches
        apply_patch(base, tmp_path / "p")


class Answer(ToolInput):
    value: int


def test_registry_schema_validation_and_finish():
    m = small_model()
    reg = ToolRegistry()
    add_weight_tools(reg, WeightInspector(m), WeightEditor(m))
    add_finish_tool(reg, Answer, "done", name="submit")
    schemas = {s["name"]: s for s in reg.schemas()}
    assert {"list_parameters", "apply_edit", "rollback", "submit"} <= set(schemas)
    assert schemas["apply_edit"]["input_schema"]["additionalProperties"] is False
    bad = reg.call("apply_edit", {"op": "set", "name": "0.bias"})  # missing fields
    assert bad.is_error and "ValidationError" in bad.content
    unknown = reg.call("nope", {})
    assert unknown.is_error
    small = ToolRegistry(max_result_chars=50)
    add_weight_tools(small, WeightInspector(m))
    assert small.call("read_slice", {"name": "0.weight"}).content.endswith("chars]")  # truncated
    ok = reg.call("apply_edit", {"op": "add", "name": "0.bias", "index": "0", "value": 1.0, "reason": "r"})
    assert not ok.is_error and json.loads(ok.content)["applied"]["op"] == "add"
    reg.call("submit", {"value": 3})
    assert reg.finished and reg.final_answer == {"value": 3} and reg.terminal_tool == "submit"
    assert len(reg.transcript) == 4
