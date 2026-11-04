"""Experiment 2 — [Write / Edit] apply engineering change notices (ECN) to SDF weights without
retraining.

Each case takes a customer part network from Experiment 1 and a natural-language ECN
(設計変更指示) such as 「中心の穴の直径を2倍に広げ、Z軸方向に+5mmオフセットせよ」. The agent must
produce a network for the changed part using only weight-space operations:

* affine edits of the first / last layer (translation, axis scaling, surface offset), and
* exact CSG composition with *written* sub-networks (polygonal prisms built from ReLU max
  gadgets) for local features — adding, enlarging or shrinking holes.

Methods compared per case (all scored against the analytic CAD ground truth of the changed part):

``no_edit``          the original network (shows how big the change is)
``wsp_edit``         the agent's weight-space edit of the customer network (0 gradient steps)
``write_from_read``  read the spec from the weights, apply the ECN to the spec, *write* a fresh
                     network analytically (0 gradient steps)
``retrain``          fit a new network to the changed CAD with gradient descent (needs the CAD /
                     ground truth for the new design — usually unavailable on site)

Run:  ``uv run wsp exp2 [--agent scripted|claude]``
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from wsp.agents.base import AgentResult, Task
from wsp.agents.prompts import EXP2_SYSTEM, exp2_user
from wsp.agents.snippets.read_sdf import read_part
from wsp.experiments.common import DEFAULT_RESULTS, PartModel, env_info, fmt, load_or_fit_parts, make_agent, write_json
from wsp.geometry import sdf_ops
from wsp.geometry.mesh import (
    chamfer_distance,
    extract_mesh,
    is_watertight,
    n_components,
    sample_surface,
    volumetric_iou,
)
from wsp.geometry.parts import MM_PER_UNIT, HoleSpec, PartSpec, part_sdf_norm
from wsp.harness.sandbox import PythonSandbox, read_snippet
from wsp.harness.tools import ToolInput, ToolRegistry, add_finish_tool
from wsp.models.cad_sdf_mlp import SDFMLP, SDFTrainConfig, fit_sdf, mlp_forward_np

S = MM_PER_UNIT
TOP_CAP = 1.5  # normalised z above the workspace used to cap blind cutters

# ----------------------------------------------------------------------------------------------
# The editable weight program
# ----------------------------------------------------------------------------------------------


def layers_to_weights(layers: sdf_ops.Layers) -> dict[str, np.ndarray]:
    return {k: v for i, (w, b) in enumerate(layers) for k, v in ((f"net.{2 * i}.weight", w), (f"net.{2 * i}.bias", b))}


@dataclass
class OpRecord:
    op: str
    args: dict[str, Any]
    reason: str
    params_before: int
    params_after: int


class SDFProgram:
    """Holds the current weight program and an undo stack; every op is exact weight algebra."""

    def __init__(self, layers: sdf_ops.Layers):
        self.layers = [(w.copy(), b.copy()) for w, b in layers]
        self.history: list[OpRecord] = []
        self._undo: list[sdf_ops.Layers] = []

    def apply(self, op: str, new_layers: sdf_ops.Layers, args: dict[str, Any], reason: str) -> dict[str, Any]:
        before = sdf_ops.param_count(self.layers)
        self._undo.append(self.layers)
        self.layers = new_layers
        rec = OpRecord(op, args, reason, before, sdf_ops.param_count(new_layers))
        self.history.append(rec)
        return {"applied": op, **self.summary()}

    def undo(self) -> dict[str, Any]:
        if not self._undo:
            raise ValueError("nothing to undo")
        self.layers = self._undo.pop()
        self.history.pop()
        return {"undone": True, **self.summary()}

    def summary(self) -> dict[str, Any]:
        return {
            "layer_shapes": [list(w.shape) for w, _ in self.layers],
            "n_params": sdf_ops.param_count(self.layers),
            "n_ops": len(self.history),
        }

    def sdf_norm(self, p: np.ndarray) -> np.ndarray:
        return mlp_forward_np(self.layers, p)


# ----------------------------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------------------------

META = {"mm_per_unit": S, "input": "xyz normalised (1 unit = 50 mm)", "output": "signed distance (normalised)"}


class NoInput(ToolInput):
    pass


class RunPythonInput(ToolInput):
    code: str = Field(
        description="numpy code over `weights` (current network, dict name->array) and `meta`. Set RESULT to "
        "return data; put same-shape replacement tensors in EDITS[name] (apply with apply_sandbox_edits)."
    )


class Reason(ToolInput):
    reason: str = Field(description="Why (kept in the audit log).")


class TranslateInput(Reason):
    dx: float = Field(0.0, description="mm")
    dy: float = Field(0.0, description="mm")
    dz: float = Field(0.0, description="mm")


class ScaleAxisInput(Reason):
    axis: Literal["x", "y", "z"]
    factor: float = Field(gt=0.05, lt=20)
    pivot: float = Field(description="Coordinate (mm) of the plane that stays fixed.")


class OffsetInput(Reason):
    delta: float = Field(description="Outward surface offset in mm (negative shrinks).", ge=-10, le=10)


class CutHoleInput(Reason):
    cx: float = Field(description="mm")
    cy: float = Field(description="mm")
    diameter: float = Field(gt=0, description="mm")
    depth: float | None = Field(None, description="Blind depth from z_top (mm); omit for a through hole.")
    z_top: float | None = Field(None, description="Top-face Z (mm); required for blind holes.")
    n_sides: int = Field(64, ge=8, le=256, description="Polygon resolution of the written prism.")


class FillCylinderInput(Reason):
    cx: float
    cy: float
    diameter: float = Field(gt=0)
    z_bottom: float
    z_top: float
    n_sides: int = Field(64, ge=8, le=256)


class FinishInput(ToolInput):
    summary: str = Field(description="What was changed and how it was verified.")


def build_ecn_registry(prog: SDFProgram) -> ToolRegistry:
    reg = ToolRegistry()
    last: dict[str, Any] = {}

    def sandbox() -> PythonSandbox:
        return PythonSandbox(layers_to_weights(prog.layers), META, timeout_s=180)

    reg.add(
        "network_summary", "Layer shapes and parameter count of the current network.", NoInput, lambda _: prog.summary()
    )

    def run_python(a: RunPythonInput) -> dict[str, Any]:
        res = sandbox().run(a.code)
        last["edits"] = res.edits
        return res.summary()

    reg.add("run_python", "Analyse the current weights in the isolated numpy sandbox.", RunPythonInput, run_python)

    def read_part_tool(_: NoInput) -> dict[str, Any]:
        res = sandbox().run(read_snippet("read_sdf"))
        if not res.ok:
            raise ValueError(res.error or "read failed")
        return res.result

    reg.add(
        "read_part",
        "Run the packaged SDF reader over the current weights (family, dimensions, holes in mm).",
        NoInput,
        read_part_tool,
    )

    def apply_sandbox_edits(a: Reason) -> dict[str, Any]:
        edits = last.get("edits") or {}
        if not edits:
            raise ValueError("the last run_python call proposed no EDITS")
        new = [(w.copy(), b.copy()) for w, b in prog.layers]
        for name, arr in edits.items():
            m = re.fullmatch(r"net\.(\d+)\.(weight|bias)", name)
            if not m or int(m.group(1)) % 2:
                raise ValueError(f"unknown tensor {name}")
            i = int(m.group(1)) // 2
            w, b = new[i]
            cur = w if m.group(2) == "weight" else b
            if arr.shape != cur.shape:
                raise ValueError(f"{name}: shape {arr.shape} != {cur.shape} (use the CSG tools for structural changes)")
            new[i] = (arr.copy(), b) if m.group(2) == "weight" else (w, arr.copy())
        last["edits"] = {}
        return prog.apply("sandbox_edits", new, {"tensors": sorted(edits)}, a.reason)

    reg.add(
        "apply_sandbox_edits",
        "Apply same-shape tensors proposed in EDITS by the last run_python call.",
        Reason,
        apply_sandbox_edits,
    )

    reg.add(
        "translate",
        "Move the whole part by (dx, dy, dz) mm. Implemented as b1 <- b1 - W1·t on the first layer.",
        TranslateInput,
        lambda a: prog.apply(
            "translate", sdf_ops.translate(prog.layers, np.array([a.dx, a.dy, a.dz]) / S), a.model_dump(), a.reason
        ),
    )
    axis_idx = {"x": 0, "y": 1, "z": 2}
    reg.add(
        "scale_axis",
        "Stretch the part along one axis about a fixed plane (first-layer W1·A, b1 + W1·c).",
        ScaleAxisInput,
        lambda a: prog.apply(
            "scale_axis",
            sdf_ops.scale_axis(prog.layers, axis_idx[a.axis], a.factor, a.pivot / S),
            a.model_dump(),
            a.reason,
        ),
    )
    reg.add(
        "offset_surface",
        "Offset the whole surface outward by delta mm (last-layer bias b_L - delta).",
        OffsetInput,
        lambda a: prog.apply(
            "offset_surface", sdf_ops.offset_surface(prog.layers, a.delta / S), a.model_dump(), a.reason
        ),
    )

    def cut_hole(a: CutHoleInput) -> dict[str, Any]:
        if a.depth is not None and a.z_top is None:
            raise ValueError("blind holes need z_top")
        z_range = None if a.depth is None else ((a.z_top - a.depth) / S, TOP_CAP)
        cutter = sdf_ops.prism(a.cx / S, a.cy / S, a.diameter / 2 / S, z_range, a.n_sides)
        return prog.apply("cut_hole", sdf_ops.subtract(prog.layers, cutter), a.model_dump(), a.reason)

    reg.add(
        "cut_hole",
        "Subtract a cylindrical hole: writes an exact ReLU prism network (n_sides-gon, area-matched) and "
        "composes max(f, -prism) with the current network.",
        CutHoleInput,
        cut_hole,
    )

    def fill_cylinder(a: FillCylinderInput) -> dict[str, Any]:
        plug = sdf_ops.prism(a.cx / S, a.cy / S, a.diameter / 2 / S, (a.z_bottom / S, a.z_top / S), a.n_sides)
        return prog.apply("fill_cylinder", sdf_ops.union(prog.layers, plug), a.model_dump(), a.reason)

    reg.add(
        "fill_cylinder",
        "Add material: union of the current part with a capped cylinder "
        "(e.g. to plug a hole before re-drilling it smaller).",
        FillCylinderInput,
        fill_cylinder,
    )
    reg.add("undo", "Undo the last operation.", NoInput, lambda _: prog.undo())
    add_finish_tool(reg, FinishInput, "Finish: the network now represents the changed part.", name="finish")
    return reg


# ----------------------------------------------------------------------------------------------
# ECN language → operations (used by the scripted agent and by write_from_read)
# ----------------------------------------------------------------------------------------------

NUM = r"([+\-−]?\d+(?:\.\d+)?)"


def _f(x: str) -> float:
    return float(x.replace("−", "-"))


def parse_ecn(text: str) -> list[dict[str, Any]]:
    """Deterministic parser for the ECN phrasings used in this experiment (the scripted stand-in for
    an LLM's language understanding). Local feature ops are returned before global transforms so
    that references such as "the centre hole" are resolved in the original coordinates."""
    local: list[dict[str, Any]] = []
    glob: list[dict[str, Any]] = []
    for m in re.finditer(rf"([XYZ])(?:軸)?方向に\s*{NUM}\s*mm\s*(?:オフセット|移動)", text):
        d = {"dx": 0.0, "dy": 0.0, "dz": 0.0}
        d[f"d{m.group(1).lower()}"] = _f(m.group(2))
        glob.append({"op": "translate", **d})
    if m := re.search(r"穴の直径を\s*(\d+(?:\.\d+)?)\s*倍", text):
        local.append({"op": "scale_main_hole", "factor": _f(m.group(1))})
    if m := re.search(r"穴径を\s*[φΦ]?\s*(\d+(?:\.\d+)?)\s*(?:mm)?\s*に", text):
        local.append({"op": "set_main_hole_diameter", "diameter": _f(m.group(1))})
    for m in re.finditer(
        rf"[(（]\s*x\s*=\s*{NUM}\s*,\s*y\s*=\s*{NUM}\s*[)）]\s*に\s*[φΦ]\s*(\d+(?:\.\d+)?)\s*の貫通穴を追加", text
    ):
        local.append({"op": "add_through_hole", "cx": _f(m.group(1)), "cy": _f(m.group(2)), "diameter": _f(m.group(3))})
    if m := re.search(r"板厚を\s*(\d+(?:\.\d+)?)\s*mm\s*に", text):
        pivot = "top" if "上面基準" in text else "bottom"
        glob.append({"op": "set_thickness", "thickness": _f(m.group(1)), "pivot": pivot})
    if (m := re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:の)?(?:肉盛り|厚く)", text)) and ("外形" in text or "肉盛" in text):
        glob.append({"op": "offset", "delta": _f(m.group(1))})
    return local + glob


def main_hole(reading: dict[str, Any]) -> dict[str, Any]:
    if not reading.get("holes"):
        raise ValueError("ECN refers to a hole but none was found in the part")
    return max(reading["holes"], key=lambda h: h["radius"])


def ops_to_tool_calls(ops: list[dict[str, Any]], reading: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Resolve ECN ops against a reading of the *original* part into concrete tool calls."""
    calls: list[tuple[str, dict[str, Any]]] = []
    z_top, z_bot = reading["z_top"], reading["z_bottom"]
    for op in ops:
        kind = op["op"]
        if kind == "translate":
            calls.append(
                (
                    "translate",
                    {"dx": op["dx"], "dy": op["dy"], "dz": op["dz"], "reason": "ECN: rigid offset of the whole part"},
                )
            )
        elif kind in ("scale_main_hole", "set_main_hole_diameter"):
            h = main_hole(reading)
            d_old = 2 * h["radius"]
            d_new = d_old * op["factor"] if kind == "scale_main_hole" else op["diameter"]
            depth = None if h["through"] else h["depth"]
            if d_new < d_old:
                calls.append(
                    (
                        "fill_cylinder",
                        {
                            "cx": h["cx"],
                            "cy": h["cy"],
                            "diameter": d_old + 2.0,
                            "z_bottom": z_bot,
                            "z_top": z_top,
                            "reason": f"ECN: plug Ø{d_old:.2f} hole before re-drilling smaller",
                        },
                    )
                )
            calls.append(
                (
                    "cut_hole",
                    {
                        "cx": h["cx"],
                        "cy": h["cy"],
                        "diameter": d_new,
                        "depth": depth,
                        "z_top": z_top,
                        "reason": f"ECN: main hole Ø{d_old:.2f} → Ø{d_new:.2f}",
                    },
                )
            )
        elif kind == "add_through_hole":
            calls.append(
                (
                    "cut_hole",
                    {"cx": op["cx"], "cy": op["cy"], "diameter": op["diameter"], "reason": "ECN: new through hole"},
                )
            )
        elif kind == "set_thickness":
            factor = op["thickness"] / (z_top - z_bot)
            pivot = z_bot if op["pivot"] == "bottom" else z_top
            calls.append(
                (
                    "scale_axis",
                    {
                        "axis": "z",
                        "factor": factor,
                        "pivot": pivot,
                        "reason": f"ECN: thickness {z_top - z_bot:.2f} → {op['thickness']} mm",
                    },
                )
            )
        elif kind == "offset":
            calls.append(("offset_surface", {"delta": op["delta"], "reason": "ECN: offset (肉盛り) of all faces"}))
        else:  # pragma: no cover
            raise ValueError(kind)
    return calls


def reading_to_spec(reading: dict[str, Any], name: str = "read") -> PartSpec:
    holes = tuple(
        HoleSpec(
            h["cx"], h["cy"], h["radius"], None if h["through"] else h["depth"], h.get("cb_radius"), h.get("cb_depth")
        )
        for h in reading.get("holes", [])
    )
    return PartSpec(
        name,
        reading["outline"],
        reading["size_x"],
        reading["size_y"],
        reading["thickness"],
        reading["z_bottom"],
        reading.get("center_x", 0.0),
        reading.get("center_y", 0.0),
        holes,
    )


def apply_ops_to_spec(spec: PartSpec, ops: list[dict[str, Any]]) -> tuple[PartSpec, float]:
    """Apply ECN ops to a parametric spec. Returns (new_spec, surface_offset_mm)."""
    offset = 0.0
    for op in ops:
        kind = op["op"]
        if kind == "translate":
            dx, dy, dz = op["dx"], op["dy"], op["dz"]
            spec = replace(
                spec,
                center_x=spec.center_x + dx,
                center_y=spec.center_y + dy,
                z_bottom=spec.z_bottom + dz,
                holes=tuple(replace(h, cx=h.cx + dx, cy=h.cy + dy) for h in spec.holes),
            )
        elif kind in ("scale_main_hole", "set_main_hole_diameter"):
            k = max(range(len(spec.holes)), key=lambda i: spec.holes[i].radius)
            h = spec.holes[k]
            r = h.radius * op["factor"] if kind == "scale_main_hole" else op["diameter"] / 2
            spec = replace(spec, holes=(*spec.holes[:k], replace(h, radius=r), *spec.holes[k + 1 :]))
        elif kind == "add_through_hole":
            spec = replace(spec, holes=(*spec.holes, HoleSpec(op["cx"], op["cy"], op["diameter"] / 2)))
        elif kind == "set_thickness":
            if op["pivot"] == "top":
                spec = replace(spec, z_bottom=spec.z_top - op["thickness"], thickness=op["thickness"])
            else:
                spec = replace(spec, thickness=op["thickness"])
        elif kind == "offset":
            offset += op["delta"]
    return spec, offset


def write_part_layers(spec: PartSpec, offset_mm: float = 0.0, n_sides: int = 64) -> sdf_ops.Layers:
    """[Write] Build an SDF network for a parametric part directly from its spec (no gradients)."""
    if spec.outline == "rect":
        lo, hi = spec.bbox_mm()
        layers = sdf_ops.box(lo / S, hi / S)
    else:
        layers = sdf_ops.prism(
            spec.center_x / S, spec.center_y / S, spec.size_x / 2 / S, (spec.z_bottom / S, spec.z_top / S), n_sides
        )
    for h in spec.holes:
        z_rng = None if h.is_through else ((spec.z_top - h.depth) / S, TOP_CAP)
        layers = sdf_ops.subtract(layers, sdf_ops.prism(h.cx / S, h.cy / S, h.radius / S, z_rng, n_sides))
        if h.has_counterbore:
            cb = sdf_ops.prism(h.cx / S, h.cy / S, h.cb_radius / S, ((spec.z_top - h.cb_depth) / S, TOP_CAP), n_sides)
            layers = sdf_ops.subtract(layers, cb)
    return sdf_ops.offset_surface(layers, offset_mm / S) if offset_mm else layers


# ----------------------------------------------------------------------------------------------
# Cases and evaluation
# ----------------------------------------------------------------------------------------------


@dataclass
class ECNCase:
    case_id: str
    part: str
    text: str
    gt: Callable[[PartSpec], tuple[PartSpec, float]]  # -> (changed spec, surface offset mm)


def _main(spec: PartSpec) -> HoleSpec:
    return max(spec.holes, key=lambda h: h.radius)


def _with_main(spec: PartSpec, **kw: Any) -> PartSpec:
    k = max(range(len(spec.holes)), key=lambda i: spec.holes[i].radius)
    return replace(spec, holes=(*spec.holes[:k], replace(spec.holes[k], **kw), *spec.holes[k + 1 :]))


ECN_CASES: list[ECNCase] = [
    ECNCase(
        "ecn1_translate_z",
        "through_hole_0",
        "部品全体をZ方向に+5mmオフセットせよ。",
        lambda s: (replace(s, z_bottom=s.z_bottom + 5), 0.0),
    ),
    ECNCase(
        "ecn2_readme_example",
        "through_hole_0",
        "中心の穴の直径を2倍に広げ、Z軸方向に+5mmオフセットせよ。",
        lambda s: (replace(_with_main(s, radius=2 * _main(s).radius), z_bottom=s.z_bottom + 5), 0.0),
    ),
    ECNCase(
        "ecn3_thickness", "plate_1", "板厚を15mmに変更せよ（下面基準）。", lambda s: (replace(s, thickness=15.0), 0.0)
    ),
    ECNCase("ecn4_offset_flange", "flange_0", "外形全体を0.5mm肉盛り（オフセット）せよ。", lambda s: (s, 0.5)),
    ECNCase(
        "ecn5_add_hole",
        "plate_0",
        "(x=+15, y=-10) にφ8の貫通穴を追加せよ。",
        lambda s: (replace(s, holes=(*s.holes, HoleSpec(15.0, -10.0, 4.0))), 0.0),
    ),
    ECNCase("ecn6_shrink_hole", "through_hole_1", "穴径をφ10に縮小せよ。", lambda s: (_with_main(s, radius=5.0), 0.0)),
    ECNCase(
        "ecn7_enlarge_flange_bore",
        "flange_1",
        "中心の穴の直径を1.5倍に広げよ。",
        lambda s: (_with_main(s, radius=1.5 * _main(s).radius), 0.0),
    ),
]


def gt_sdf(spec: PartSpec, offset_mm: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda p: part_sdf_norm(spec, p) - offset_mm / S


def evaluate_network(
    layers: sdf_ops.Layers, spec: PartSpec, offset_mm: float, res_mm: float = 0.5, seed: int = 0
) -> dict[str, Any]:
    """Geometry metrics of a weight program against the changed CAD ground truth."""
    lo, hi = spec.bbox_mm()
    lo, hi = (lo - 6.0 - offset_mm) / S, (hi + 6.0 + offset_mm) / S
    f_net = SDFMLP.from_layers(layers).sdf_numpy  # float32 torch forward (fast); exact algebra is float64
    f_gt = gt_sdf(spec, offset_mm)
    m_net = extract_mesh(f_net, lo, hi, res_mm / S)
    m_gt = extract_mesh(f_gt, lo, hi, res_mm / S)
    rng = np.random.default_rng(seed)
    cd = chamfer_distance(sample_surface(m_net, 60_000, rng) * S, sample_surface(m_gt, 60_000, rng) * S)
    out: dict[str, Any] = {
        "chamfer_mm": cd["chamfer"],
        "p95_mm": cd["p95"],
        "iou": volumetric_iou(f_net, f_gt, lo, hi, 2 * res_mm / S),
        "watertight": is_watertight(m_net),
        "components": n_components(m_net),
        "n_params": sdf_ops.param_count(layers),
        "depth": len(layers),
    }
    # Dimensional read-back with the same reader the agent uses (closed loop Read → Write → Read).
    # A surface offset δ grows the thickness by 2δ, lowers the bottom face by δ and shrinks holes by δ.
    r = read_part(layers, S)
    if not r.get("empty"):
        out["readback"] = {
            "family": r["family"],
            "thickness_err": abs(r["thickness"] - (spec.thickness + 2 * offset_mm)),
            "z_bottom_err": abs(r["z_bottom"] - (spec.z_bottom - offset_mm)),
            "n_holes": len(r["holes"]),
            "n_holes_gt": len(spec.holes),
        }
        if spec.holes and r["holes"]:
            gt = _main(spec)
            rh = min(r["holes"], key=lambda h: np.hypot(h["cx"] - gt.cx, h["cy"] - gt.cy))
            out["readback"]["main_hole_diameter_err"] = abs(2 * rh["radius"] - 2 * (gt.radius - offset_mm))
            out["readback"]["main_hole_center_err"] = float(np.hypot(rh["cx"] - gt.cx, rh["cy"] - gt.cy))
    return out


def run_case(case: ECNCase, pm: PartModel, agent: Any, retrain_cfg: SDFTrainConfig, do_retrain: bool) -> dict[str, Any]:
    from wsp.agents.scripted import ScriptedECNEditor

    spec_new, offset = case.gt(pm.spec)
    base_layers = pm.model.to_layers()
    row: dict[str, Any] = {
        "case": case.case_id,
        "part": case.part,
        "ecn": case.text,
        "gt_spec": spec_new.to_dict(),
        "gt_offset_mm": offset,
        "methods": {},
    }

    row["methods"]["no_edit"] = {"gradient_steps": 0, "seconds": 0.0, **evaluate_network(base_layers, spec_new, offset)}

    # --- agent edit -----------------------------------------------------------------------------
    prog = SDFProgram(base_layers)
    reg = build_ecn_registry(prog)
    task = Task(case.case_id, EXP2_SYSTEM, exp2_user(case.case_id, case.text), context={"ecn": case.text})
    t0 = time.perf_counter()
    result: AgentResult = (agent or ScriptedECNEditor()).run(task, reg)
    secs = time.perf_counter() - t0
    row["methods"]["wsp_edit"] = {
        "gradient_steps": 0,
        "seconds": secs,
        "ops": [r.__dict__ for r in prog.history],
        "agent": result.to_dict(),
        **evaluate_network(prog.layers, spec_new, offset),
    }

    # --- write from read --------------------------------------------------------------------------
    t0 = time.perf_counter()
    reading = read_part(base_layers, S)
    spec_read, off_read = apply_ops_to_spec(reading_to_spec(reading), parse_ecn(case.text))
    written = write_part_layers(spec_read, off_read)
    row["methods"]["write_from_read"] = {
        "gradient_steps": 0,
        "seconds": time.perf_counter() - t0,
        **evaluate_network(written, spec_new, offset),
    }

    # --- retraining baseline ----------------------------------------------------------------------
    if do_retrain:
        lo, hi = spec_new.bbox_mm()
        model, fit = fit_sdf(gt_sdf(spec_new, offset), ((lo - offset) / S, (hi + offset) / S), retrain_cfg)
        row["methods"]["retrain"] = {
            "gradient_steps": fit.gradient_steps,
            "seconds": fit.seconds,
            **evaluate_network(model.to_layers(), spec_new, offset),
        }
    return row


METHODS = ("no_edit", "wsp_edit", "write_from_read", "retrain")


def to_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Experiment 2 — ECN edits in weight space (agent: `{summary['agent']}`)",
        "",
        "Chamfer distance / P95 in mm against the analytic CAD of the *changed* part (lower is better). "
        "`grad` = gradient steps used. Read-back = the Exp. 1 reader run on the resulting network.",
        "",
        "| case | ECN | method | Chamfer | P95 | IoU | watertight | read-back Ø err | read-back t err | grad | sec |",
        "|---|---|---|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        for m in METHODS:
            if m not in r["methods"]:
                continue
            x = r["methods"][m]
            rb = x.get("readback", {})
            lines.append(
                f"| {r['case']} | {r['ecn'] if m == 'no_edit' else ''} | {m} | {fmt(x['chamfer_mm'], 3)} | "
                f"{fmt(x['p95_mm'], 2)} | "
                f"{fmt(x['iou'], 3)} | {'✅' if x['watertight'] else '❌'} | {fmt(rb.get('main_hole_diameter_err'))} | "
                f"{fmt(rb.get('thickness_err'))} | {x['gradient_steps']} | {fmt(x['seconds'], 1)} |"
            )
    a = summary["aggregate"]
    lines += [
        "",
        "## Mean over cases",
        "",
        "| method | Chamfer [mm] | P95 [mm] | IoU | watertight | gradient steps | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, v in a.items():
        lines.append(
            f"| {m} | {fmt(v['chamfer_mm'], 3)} | {fmt(v['p95_mm'], 2)} | {fmt(v['iou'], 3)} | "
            f"{v['watertight_rate']:.0%} | {v['gradient_steps']:.0f} | {fmt(v['seconds'], 1)} |"
        )
    return "\n".join(lines) + "\n"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for m in METHODS:
        xs = [r["methods"][m] for r in rows if m in r["methods"]]
        if not xs:
            continue
        out[m] = {
            "chamfer_mm": float(np.mean([x["chamfer_mm"] for x in xs])),
            "p95_mm": float(np.mean([x["p95_mm"] for x in xs])),
            "iou": float(np.mean([x["iou"] for x in xs])),
            "watertight_rate": float(np.mean([x["watertight"] for x in xs])),
            "gradient_steps": float(np.mean([x["gradient_steps"] for x in xs])),
            "seconds": float(np.mean([x["seconds"] for x in xs])),
        }
    return out


def run(
    agent_name: str = "scripted",
    seed: int = 0,
    out_dir: Path | None = None,
    retrain: bool = True,
    cases: list[str] | None = None,
    **agent_kwargs: Any,
) -> dict[str, Any]:
    out_dir = Path(out_dir or DEFAULT_RESULTS / "exp2")
    agent = make_agent(agent_name, **agent_kwargs)
    parts = {pm.spec.name: pm for pm in load_or_fit_parts(2, seed)}
    rows = []
    for case in ECN_CASES:
        if cases and case.case_id not in cases:
            continue
        print(f"[exp2] {case.case_id}: {case.text}", flush=True)
        row = run_case(case, parts[case.part], agent, SDFTrainConfig(seed=seed + 1), retrain)
        rows.append(row)
        w = row["methods"]["wsp_edit"]
        print(
            f"   wsp_edit chamfer {w['chamfer_mm']:.3f} mm (no_edit {row['methods']['no_edit']['chamfer_mm']:.3f}), "
            f"ops={[o['op'] for o in w['ops']]}"
        )
    summary = {
        "experiment": "exp2_write_ecn_edit",
        "agent": agent_name if agent is None else agent.name,
        "seed": seed,
        "aggregate": aggregate(rows),
        "env": env_info(),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "cases.json", rows)
    (out_dir / "README.md").write_text(to_markdown(summary, rows))
    print(f"[exp2] done → {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="scripted", choices=["scripted", "claude"])
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-retrain", action="store_true", help="skip the gradient-descent baseline")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args(argv)
    kw = {"model": a.model} if a.model else {}
    run(a.agent, a.seed, a.out_dir, retrain=not a.no_retrain, cases=a.cases, **kw)


if __name__ == "__main__":
    main()
