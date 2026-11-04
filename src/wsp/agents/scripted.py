"""Deterministic reference agents.

They interact with the model **only through the same tool registry the LLM agent gets** — every
read, probe and edit goes through validation, the forward-pass budget and the audit transcript.
They serve three purposes: offline/CI runs without an API key, a reproducible baseline to compare
LLM agents against, and an executable specification of a sensible weight-space workflow.

The policies encode the reasoning steps explicitly (diagnose → hypothesise → minimal edit →
verify); an LLM agent is expected to discover equivalent steps from the prompt and tool results.
"""

from __future__ import annotations

import numpy as np

from wsp.agents.base import AgentResult, Task, result_from_registry
from wsp.harness.sandbox import read_snippet
from wsp.harness.tools import ToolRegistry

# ----------------------------------------------------------------------------------------------
# Experiment 1: read a part from SDF weights
# ----------------------------------------------------------------------------------------------


class ScriptedReader:
    name = "scripted"

    def run(self, task: Task, tools: ToolRegistry) -> AgentResult:
        params = tools.call_json("list_parameters")
        first = params[0]["name"]
        stats = tools.call_json("tensor_stats", {"name": first})
        out = tools.call_json("run_python", {"code": read_snippet("read_sdf")})
        if not out["ok"] or out["result"] is None:
            return result_from_registry(self.name, tools, "analysis_failed", notes=[str(out.get("error"))])
        r = out["result"]
        if r.get("empty"):
            return result_from_registry(self.name, tools, "empty_field", notes=["network has no interior (f >= 0)"])

        holes = [
            {k: h[k] for k in ("cx", "cy", "radius", "through", "depth", "cb_radius", "cb_depth")} for h in r["holes"]
        ]
        evidence = [
            f"architecture {r['architecture']['shapes']} ({r['architecture']['n_params']} params) decoded from "
            f"{len(params)} tensors; first-layer column norms (x, y, z) = {stats.get('col_norms', '?')}",
            f"top-view footprint fill ratio {r['footprint_fill_ratio']:.3f} → outline '{r['outline']}' "
            "(rectangle ≈ 1.0, disk ≈ π/4 ≈ 0.785)",
            f"thickness {r['thickness']:.2f} mm: median of Z bisections along typical full-thickness columns",
        ]
        for i, h in enumerate(r["holes"]):
            kind = "through" if h["through"] else f"blind, depth {h['depth']:.2f} mm (material found on the axis)"
            if h["cb_radius"]:
                cb_depth = h["cb_depth"] or float("nan")
                kind += f", counterbore R{h['cb_radius']:.2f} depth {cb_depth:.2f} mm (radius step along Z)"
            evidence.append(
                f"hole {i}: empty region enclosed by material below the top face; circle fit to bisected "
                f"boundary points → centre ({h['cx']:.2f}, {h['cy']:.2f}), R{h['radius']:.2f} mm "
                f"(rms {h['circle_rms_mm']:.3f}); {kind}"
            )
        if not r["holes"]:
            evidence.append("no enclosed empty region in the top-view slice → no holes")
        evidence.append(f"{r['n_eval']} field evaluations of the decoded program (numpy, in the sandbox)")

        tools.call_json(
            "submit_reading",
            {
                "family": r["family"],
                "outline": r["outline"],
                "size_x": r["size_x"],
                "size_y": r["size_y"],
                "thickness": r["thickness"],
                "z_bottom": r["z_bottom"],
                "holes": holes,
                "rationale": "\n".join(evidence),
            },
        )
        return result_from_registry(self.name, tools, "finished")


# ----------------------------------------------------------------------------------------------
# Experiment 2: apply an ECN by weight-space operations
# ----------------------------------------------------------------------------------------------


class ScriptedECNEditor:
    """Parse the ECN → resolve references by reading the part → apply exact weight ops → verify by
    reading again; holes that came out undersized (e.g. a global offset on a field that is not a true
    distance near the hole, |∇f| < 1) are repaired locally by re-cutting them to the expected size."""

    name = "scripted"

    def __init__(self, repair_tol_mm: float = 0.1):
        self.repair_tol_mm = repair_tol_mm

    def run(self, task: Task, tools: ToolRegistry) -> AgentResult:
        from wsp.experiments.exp2_write_ecn_edit import apply_ops_to_spec, ops_to_tool_calls, parse_ecn, reading_to_spec

        ecn = task.context["ecn"]
        ops = parse_ecn(ecn)
        notes = [f"parsed ECN into ops: {ops}"]
        if not ops:
            return result_from_registry(self.name, tools, "ecn_not_understood", notes=notes)
        before = tools.call_json("read_part")
        for name, args in ops_to_tool_calls(ops, before):
            tools.call_json(name, args)
        after = tools.call_json("read_part")

        # verify-and-repair against the expected spec (reading of the original part + ECN)
        expected, offset = apply_ops_to_spec(reading_to_spec(before), ops)
        repaired = False
        for h in expected.holes:
            if not after["holes"]:
                break
            got = min(after["holes"], key=lambda g: np.hypot(g["cx"] - h.cx, g["cy"] - h.cy))
            d_exp, d_got = 2 * (h.radius - offset), 2 * got["radius"]
            if d_got < d_exp - self.repair_tol_mm:
                notes.append(
                    f"verify: hole at ({h.cx:.1f}, {h.cy:.1f}) is Ø{d_got:.2f}, expected Ø{d_exp:.2f} → re-cut"
                )
                depth = None if h.is_through else h.depth - offset
                tools.call_json(
                    "cut_hole",
                    {
                        "cx": h.cx,
                        "cy": h.cy,
                        "diameter": d_exp,
                        "depth": depth,
                        "z_top": expected.z_top + offset,
                        "reason": "verify-and-repair: restore expected hole size",
                    },
                )
                repaired = True
            elif d_got > d_exp + self.repair_tol_mm:
                notes.append(
                    f"verify: hole at ({h.cx:.1f}, {h.cy:.1f}) is Ø{d_got:.2f} > expected Ø{d_exp:.2f} (not repaired)"
                )
        t_exp = expected.thickness + 2 * offset
        if abs(after["thickness"] - t_exp) > 0.2:
            notes.append(f"verify: thickness {after['thickness']:.2f} mm differs from expected {t_exp:.2f} mm")
        if repaired:
            after = tools.call_json("read_part")
        d_before = [round(2 * h["radius"], 2) for h in before["holes"]]
        d_after = [round(2 * h["radius"], 2) for h in after["holes"]]
        notes.append(
            "verification read-back: "
            f"family {before['family']} → {after['family']}, "
            f"thickness {before['thickness']:.2f} → {after['thickness']:.2f} mm, "
            f"z_bottom {before['z_bottom']:.2f} → {after['z_bottom']:.2f} mm, holes Ø {d_before} → {d_after} mm"
        )
        tools.call_json("finish", {"summary": "; ".join(notes)})
        return result_from_registry(self.name, tools, "finished", notes=notes)


# ----------------------------------------------------------------------------------------------
# Experiment 3: reroute an existing tolerance detector into the class head
# ----------------------------------------------------------------------------------------------


class ScriptedFixer:
    """Diagnose-then-patch policy for the tolerance-blind classifier.

    1. Evaluate the few-shot set → which confusion? (リーマ → キリ)
    2. Counterfactual probes: same geometry with / without a tolerance token. If the auxiliary
       tolerance head reacts strongly but the class logits barely move, the model *can* see the
       tolerance but the class head does not use it — a routing bottleneck, not a missing skill.
    3. Read the tolerance direction straight from the ``aux_tol`` weights and find a dead fusion
       unit (never active) that can carry it without disturbing existing computations.
    4. Write the unit: input weights = tolerance direction, output weights = +α ream / −α drill.
       Calibrate α with probes (smallest α that makes every tolerance probe 'ream'), then ×2 margin.
    5. Verify in a transaction (few-shot accuracy must not drop) and submit with the evidence.
    """

    name = "scripted"

    def __init__(self, ablate_line_weight: bool = False, alphas: tuple[float, ...] = (0.25, 0.5, 1, 2, 4, 8, 16, 32)):
        self.ablate_line_weight = ablate_line_weight
        self.alphas = alphas

    def run(self, task: Task, tools: ToolRegistry) -> AgentResult:
        classes: tuple[str, ...] = task.context["classes"]
        geo_features: tuple[str, ...] = task.context["geo_features"]
        ream, drill = classes.index("ream"), classes.index("drill")
        notes: list[str] = []

        # 1. symptoms
        ev = tools.call_json("evaluate_fewshot")
        base_acc = ev["accuracy"]
        conf = np.array(ev["confusion(rows=true, cols=pred)"])
        notes.append(f"few-shot accuracy {base_acc:.3f}; ream→drill confusions {conf[ream, drill]}/{conf[ream].sum()}")
        plain = [s["i"] for s in ev["samples"] if s["label"] in ("ream", "drill")]

        # 2. causal evidence on the misclassified samples
        wrong = [x for x in ev["samples"] if x["label"] == "ream" and x["pred"] != "ream"]
        lw_name = "line_weight_mm*2"
        items = []
        if wrong:
            fired = [x["aux_tol_logit"] for x in wrong]
            notes.append(
                f"{len(wrong)} リーマ samples predicted {sorted({x['pred'] for x in wrong})} although the auxiliary "
                f"tolerance detector fires on them (aux_tol logits {np.round(fired, 1).tolist()} > 0) → the capability "
                "exists but does not reach the class decision"
            )
            # Counterfactual: identical sample, only the pen weight swapped to tenant A's thick reamer pen.
            for x in wrong:
                items += [
                    {"text": x["text"], "geometry": x["geo"]},
                    {"text": x["text"], "geometry": {**x["geo"], lw_name: 1.0}},
                ]
            pr = tools.call_json("probe", {"items": items})["items"]
            flips = sum(pr[k]["pred"] != "ream" and pr[k + 1]["pred"] == "ream" for k in range(0, len(pr), 2))
            notes.append(
                f"counterfactual: changing only the pen weight to 0.5 mm flips {flips}/{len(wrong)} of them to リーマ "
                "→ the decision is driven by line weight (template shortcut), not by the tolerance call-out"
            )
        else:
            notes.append("no misclassified リーマ in the few-shot set; proceeding with the domain-rule check only")
        items = [{"text": f"φ10{t}", "like_fewshot": i} for i in plain for t in ("", "H7")]

        # line-weight shortcut evidence from the first geometry layer
        lw = geo_features.index(lw_name)
        w_geo = np.array(tools.call_json("read_slice", {"name": "geo_enc.0.weight"})["values"])
        norms = np.linalg.norm(w_geo, axis=0)
        rank = int((norms > norms[lw]).sum()) + 1
        notes.append(
            f"geo_enc.0 input-column norms: line weight {norms[lw]:.2f} is #{rank} of {len(norms)} features "
            f"(max {norms.max():.2f}) — a template property the first layer weights heavily"
        )

        # 3. tolerance direction + a free unit
        u = np.array(tools.call_json("read_slice", {"name": "aux_tol.weight", "index": "0, :"})["values"])
        b = float(tools.call_json("read_slice", {"name": "aux_tol.bias"})["values"][0])
        acts = tools.call_json("activation_stats", {"module": "fuse_act"})["fuse_act"]
        dead = acts["dead_units"]
        # Make sure the candidate is also silent on the tolerance probes (not just on the few-shot set).
        pr_c = tools.call_json("probe", {"items": items[:8], "capture": ["fuse_act"]})["items"]
        active_on_probes = set(np.nonzero(np.max([p["fuse_act"] for p in pr_c], axis=0) > 0)[0].tolist())
        free = [j for j in dead if j not in active_on_probes]
        if free:
            j = free[0]
            notes.append(f"dead fusion units {dead}; using unit {j} (silent on few-shot and probes)")
        else:
            j = int(np.argmin(acts["frac_active"]))
            notes.append(f"no dead unit; repurposing least-active unit {j} (frac_active {acts['frac_active'][j]})")
        d = len(u)

        # 4. write the new pathway inside a checkpoint so it can be rolled back
        tools.call_json("checkpoint", {"label": "before tolerance reroute"})
        why = "route existing tolerance detector (aux_tol direction) into fusion unit"
        tools.call_json(
            "apply_edit",
            {
                "op": "set",
                "name": "fuse.weight",
                "index": f"{j}, 0:{d}",
                "value": 0.0,
                "reason": f"{why}: disconnect geometry inputs of unit {j}",
            },
        )
        tools.call_json(
            "apply_edit",
            {
                "op": "set",
                "name": "fuse.weight",
                "index": f"{j}, {d}:{2 * d}",
                "value": u.tolist(),
                "reason": f"{why}: text inputs = aux_tol.weight",
            },
        )
        tools.call_json(
            "apply_edit",
            {
                "op": "set",
                "name": "fuse.bias",
                "index": f"{j}",
                "value": b,
                "reason": f"{why}: bias = aux_tol.bias, so unit = relu(tolerance logit)",
            },
        )
        tools.call_json(
            "apply_edit",
            {
                "op": "set",
                "name": "cls.weight",
                "index": f":, {j}",
                "value": 0.0,
                "reason": "clear previous outgoing weights of the repurposed unit",
            },
        )

        # Calibration probes: every fit-tolerance code the tokenizer knows × typical phrasings, on the
        # geometry of 2 drill + 2 ream few-shot holes. Require a margin (p(ream) ≥ 0.9), not just argmax.
        # Only codes the existing detector actually recognises can be rerouted; the rest would need a
        # new capability (cf. "fixing bottlenecks works, writing new skills is hard").
        all_codes = task.context.get("tolerance_tokens") or ["H7"]
        det = tools.call_json("probe", {"items": [{"text": f"φ10{t}", "like_fewshot": plain[0]} for t in all_codes]})
        tol_tokens = [t for t, p in zip(all_codes, det["items"], strict=True) if p["aux_tol_logit"] > 1.0]
        unknown = [t for t in all_codes if t not in tol_tokens]
        notes.append(
            f"tolerance detector recognises {tol_tokens}; not recognised (no capability to reroute): {unknown}"
        )
        geo_ids = [s["i"] for s in ev["samples"] if s["label"] == "drill"][:2] + [
            s["i"] for s in ev["samples"] if s["label"] == "ream"
        ][:2]
        templates = ("φ10{t}", "4-φ10{t} 通し", "φ10{t} THRU")
        tol_items = [
            {"text": tpl.format(t=t), "like_fewshot": i} for i in geo_ids for t in tol_tokens for tpl in templates
        ]
        chosen = None
        for alpha in self.alphas:
            self._set_alpha(tools, j, ream, drill, alpha)
            probes = tools.call_json("probe", {"items": tol_items})["items"]
            min_p = min(p["probs"]["ream"] for p in probes)
            acc = tools.call_json("evaluate_fewshot")["accuracy"]
            if min_p >= 0.9 and acc >= base_acc:
                chosen = alpha
                notes.append(
                    f"α={alpha}: min p(ream) {min_p:.3f} over {len(tol_items)} tolerance probes "
                    f"({len(tol_tokens)} codes × {len(templates)} phrasings), few-shot acc {acc:.3f}"
                )
                break
        if chosen is None:
            tools.call_json("rollback", {})
            notes.append("no α satisfied the checks — rolled back, no fix applied")
            tools.call_json(
                "submit_fix",
                {
                    "diagnosis": "; ".join(notes),
                    "edit_description": "none (rolled back)",
                    "expected_side_effects": "none",
                },
            )
            return result_from_registry(self.name, tools, "rolled_back", notes=notes)
        final_alpha = 2 * chosen
        self._set_alpha(tools, j, ream, drill, final_alpha)
        notes.append(f"final α = 2 × {chosen} = {final_alpha} (safety margin)")

        if self.ablate_line_weight:
            fs = tools.call_json("evaluate_fewshot")
            mean_lw = float(np.mean([s["geo"]["line_weight_mm*2"] for s in fs["samples"]]))  # tenant-B typical value
            w0 = np.array(tools.call_json("read_slice", {"name": "geo_enc.0.weight", "index": f":, {lw}"})["values"])
            tools.call_json(
                "apply_edit",
                {
                    "op": "add",
                    "name": "geo_enc.0.bias",
                    "value": (w0 * mean_lw).tolist(),
                    "reason": "mean-ablate line weight: fold its typical contribution into the bias",
                },
            )
            tools.call_json(
                "apply_edit",
                {
                    "op": "set",
                    "name": "geo_enc.0.weight",
                    "index": f":, {lw}",
                    "value": 0.0,
                    "reason": "domain rule 2: pen line weight must not decide the hole type",
                },
            )
            notes.append("mean-ablated the line-weight input (tenant-specific patch)")

        final = tools.call_json("evaluate_fewshot")
        summary = tools.call_json("edit_summary")
        notes.append(
            f"few-shot accuracy {base_acc:.3f} → {final['accuracy']:.3f}; changed fraction "
            f"{summary['changed_fraction']:.4%}; budget used {final['budget']['used']}"
        )
        tools.call_json(
            "submit_fix",
            {
                "diagnosis": " | ".join(notes[:3]),
                "edit_description": (
                    f"fusion unit {j} rewritten as relu(aux_tol logit) with ±{final_alpha} to ream/drill"
                    + (" + line-weight mean ablation" if self.ablate_line_weight else "")
                ),
                "expected_side_effects": (
                    "Unit is zero whenever no tolerance token is present, so predictions without a tolerance "
                    "call-out are unchanged by construction; holes with a tolerance call-out are pushed to リーマ."
                ),
            },
        )
        return result_from_registry(self.name, tools, "finished", notes=notes)

    @staticmethod
    def _set_alpha(tools: ToolRegistry, j: int, ream: int, drill: int, alpha: float) -> None:
        tools.call_json(
            "apply_edit",
            {
                "op": "set",
                "name": "cls.weight",
                "index": f"{ream}, {j}",
                "value": alpha,
                "reason": "tolerance unit → +α on リーマ logit",
            },
        )
        tools.call_json(
            "apply_edit",
            {
                "op": "set",
                "name": "cls.weight",
                "index": f"{drill}, {j}",
                "value": -alpha,
                "reason": "tolerance unit → −α on キリ logit",
            },
        )
