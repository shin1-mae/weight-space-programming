"""Experiment 1 — [Read] recover 3D part geometry and design intent from SDF network weights.

For every generated machine part (plate / through hole / blind hole / counterbore / flange) a
small ReLU MLP is fitted to its SDF — this plays the role of the customer's model. The agent is
then given **only the weights** (plus a model card) and tools to inspect them, and must report
the part family, outer dimensions and hole parameters. Readings are scored against the CAD
ground truth.

Run:  ``uv run wsp exp1 [--agent scripted|claude]``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from wsp.agents.base import AgentResult, Task
from wsp.agents.prompts import EXP1_SYSTEM, exp1_user
from wsp.experiments.common import (
    DEFAULT_RESULTS,
    PartModel,
    env_info,
    fmt,
    load_or_fit_parts,
    make_agent,
    write_json,
)
from wsp.geometry.parts import FAMILIES, MM_PER_UNIT, PartSpec
from wsp.harness.inspector import WeightInspector
from wsp.harness.sandbox import PythonSandbox, weights_from_module
from wsp.harness.tools import ToolInput, ToolRegistry, add_finish_tool, add_weight_tools
from wsp.models.cad_sdf_mlp import SDFMLP

# ----------------------------------------------------------------------------------------------
# Answer schema (the terminal tool's input)
# ----------------------------------------------------------------------------------------------


class HoleReading(ToolInput):
    cx: float = Field(description="Hole axis X (mm).")
    cy: float = Field(description="Hole axis Y (mm).")
    radius: float = Field(description="Radius of the (inner) hole (mm).")
    through: bool = Field(description="True for a through hole.")
    depth: float | None = Field(None, description="Blind-hole depth from the top face (mm).")
    cb_radius: float | None = Field(None, description="Counterbore radius (mm), if any.")
    cb_depth: float | None = Field(None, description="Counterbore depth from the top face (mm), if any.")


class PartReading(ToolInput):
    family: Literal["plate", "through_hole", "blind_hole", "counterbore", "flange"]
    outline: Literal["rect", "disk"]
    size_x: float = Field(description="Outer size along X (mm); diameter for a disk.")
    size_y: float = Field(description="Outer size along Y (mm); diameter for a disk.")
    thickness: float = Field(description="Plate / flange thickness (mm).")
    z_bottom: float = Field(description="Z of the bottom face (mm).")
    holes: list[HoleReading] = Field(default_factory=list)
    rationale: str = Field(description="Evidence from the weights supporting the reading.")


META = {
    "mm_per_unit": MM_PER_UNIT,
    "input": "xyz normalised (1 unit = 50 mm)",
    "output": "signed distance, normalised units, negative inside",
    "activation": "relu",
}


def build_read_registry(model: SDFMLP) -> ToolRegistry:
    reg = ToolRegistry()
    weights = weights_from_module(model)
    add_weight_tools(
        reg,
        WeightInspector(model),
        editor=None,
        sandbox_factory=lambda: PythonSandbox(weights, META, timeout_s=120),
    )
    add_finish_tool(
        reg, PartReading, "Submit the final reading of the part (call exactly once).", name="submit_reading"
    )
    return reg


# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------


def score_reading(spec: PartSpec, reading: dict[str, Any] | None) -> dict[str, Any]:
    if not reading:
        return {"family_correct": False, "answered": False}
    s: dict[str, Any] = {
        "answered": True,
        "family_gt": spec.family,
        "family_pred": reading["family"],
        "family_correct": reading["family"] == spec.family,
        "size_x_err": abs(reading["size_x"] - spec.size_x),
        "size_y_err": abs(reading["size_y"] - spec.size_y),
        "thickness_err": abs(reading["thickness"] - spec.thickness),
        "z_bottom_err": abs(reading["z_bottom"] - spec.z_bottom),
        "hole_count_correct": len(reading["holes"]) == len(spec.holes),
    }
    if spec.holes and reading["holes"]:
        gt = spec.holes[0]
        rh = min(reading["holes"], key=lambda h: np.hypot(h["cx"] - gt.cx, h["cy"] - gt.cy))
        s["center_err"] = float(np.hypot(rh["cx"] - gt.cx, rh["cy"] - gt.cy))
        s["radius_err"] = abs(rh["radius"] - gt.radius)
        s["through_correct"] = bool(rh["through"]) == gt.is_through
        if gt.depth is not None and rh.get("depth") is not None:
            s["depth_err"] = abs(rh["depth"] - gt.depth)
        if gt.has_counterbore and rh.get("cb_radius") is not None:
            s["cb_radius_err"] = abs(rh["cb_radius"] - gt.cb_radius)
            if rh.get("cb_depth") is not None:
                s["cb_depth_err"] = abs(rh["cb_depth"] - gt.cb_depth)
    return s


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return float(np.mean(vals)) if vals else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "size_x_err",
        "size_y_err",
        "thickness_err",
        "z_bottom_err",
        "center_err",
        "radius_err",
        "depth_err",
        "cb_radius_err",
        "cb_depth_err",
    ]
    out: dict[str, Any] = {
        "n_parts": len(rows),
        "family_accuracy": float(np.mean([r["family_correct"] for r in rows])),
        "hole_count_accuracy": float(np.mean([r.get("hole_count_correct", False) for r in rows])),
        "through_accuracy": _mean([{"v": float(r["through_correct"])} for r in rows if "through_correct" in r], "v"),
    }
    for k in keys:
        out[f"mean_{k}_mm"] = _mean(rows, k)
    return out


# ----------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------


def run_case(pm: PartModel, agent: Any) -> tuple[AgentResult, dict[str, Any], float]:
    from wsp.agents.scripted import ScriptedReader

    reg = build_read_registry(pm.model)
    task = Task(name=pm.spec.name, system=EXP1_SYSTEM, user=exp1_user(pm.spec.name))
    t0 = time.perf_counter()
    result = (agent or ScriptedReader()).run(task, reg)
    return result, score_reading(pm.spec, result.final_answer), time.perf_counter() - t0


def to_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    a = summary["aggregate"]
    lines = [
        f"# Experiment 1 — Read 3D semantics from SDF weights (agent: `{summary['agent']}`)",
        "",
        f"- parts: {a['n_parts']} ({', '.join(FAMILIES)}), network: ReLU MLP {summary['network']}",
        f"- family accuracy: **{a['family_accuracy']:.0%}**, hole-count accuracy: {a['hole_count_accuracy']:.0%}, "
        f"through/blind accuracy: {fmt(a['through_accuracy'], 2)}",
        f"- mean abs error [mm]: size_x {fmt(a['mean_size_x_err_mm'])}, size_y {fmt(a['mean_size_y_err_mm'])}, "
        f"thickness {fmt(a['mean_thickness_err_mm'])}, z_bottom {fmt(a['mean_z_bottom_err_mm'])}, "
        f"hole centre {fmt(a['mean_center_err_mm'])}, hole radius {fmt(a['mean_radius_err_mm'])}, "
        f"blind depth {fmt(a['mean_depth_err_mm'])}, counterbore radius {fmt(a['mean_cb_radius_err_mm'])}, "
        f"counterbore depth {fmt(a['mean_cb_depth_err_mm'])}",
        "",
        "| part | GT family | read family | thickness err | hole centre err | radius err | depth / cb err "
        "| tool calls |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["score"]
        extra = s.get("depth_err", s.get("cb_radius_err"))
        lines.append(
            f"| {r['part']} | {s.get('family_gt', '?')} | {s.get('family_pred', '–')} "
            f"{'✅' if s.get('family_correct') else '❌'} | {fmt(s.get('thickness_err'))} | "
            f"{fmt(s.get('center_err'))} | {fmt(s.get('radius_err'))} | {fmt(extra)} | {r['n_tool_calls']} |"
        )
    lines.append("")
    lines.append(
        "All errors in millimetres. Ground truth comes from the parametric CAD spec the network was fitted to."
    )
    return "\n".join(lines) + "\n"


def run(
    agent_name: str = "scripted", n_per_family: int = 2, seed: int = 0, out_dir: Path | None = None, **agent_kwargs: Any
) -> dict[str, Any]:
    out_dir = Path(out_dir or DEFAULT_RESULTS / "exp1")
    agent = make_agent(agent_name, **agent_kwargs)
    print(f"[exp1] preparing {n_per_family * len(FAMILIES)} part networks (seed={seed})")
    parts = load_or_fit_parts(n_per_family, seed)
    rows = []
    for pm in parts:
        result, score, secs = run_case(pm, agent)
        rows.append(
            {
                "part": pm.spec.name,
                "spec": pm.spec.to_dict(),
                "fit_seconds": pm.fit.seconds,
                "score": score,
                "seconds": secs,
                "n_tool_calls": result.n_tool_calls,
                "agent_result": result.to_dict(),
            }
        )
        print(
            f"  {pm.spec.name:16s} → {score.get('family_pred')} ({'ok' if score['family_correct'] else 'WRONG'}), "
            f"radius err {fmt(score.get('radius_err'))} mm, {secs:.1f}s"
        )
    summary = {
        "experiment": "exp1_read_3d_semantics",
        "agent": agent_name if agent is None else agent.name,
        "seed": seed,
        "network": str(parts[0].model.hidden_sizes) if parts else "",
        "aggregate": aggregate([r["score"] for r in rows]),
        "env": env_info(),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "cases.json", rows)
    (out_dir / "README.md").write_text(to_markdown(summary, rows))
    print(f"[exp1] family accuracy {summary['aggregate']['family_accuracy']:.0%} → {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="scripted", choices=["scripted", "claude"])
    p.add_argument("--model", default=None, help="Claude model id (agent=claude)")
    p.add_argument("--n-per-family", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args(argv)
    kw = {"model": a.model} if a.model else {}
    run(a.agent, a.n_per_family, a.seed, a.out_dir, **kw)


if __name__ == "__main__":
    main()
