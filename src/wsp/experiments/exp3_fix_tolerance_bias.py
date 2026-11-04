"""Experiment 3 — [Fix] surgical hot-fix of a drawing hole classifier that ignores fit tolerances.

Scenario (see :mod:`wsp.data.drawings`): a multimodal hole-type classifier trained on tenant A
learned the shortcut "thick pen ⇒ reamed hole". At tenant B (thin pens everywhere, tolerance always
written next to the diameter) reamed holes with ``H7`` are classified as plain drill holes.

The agent may not retrain. It gets 16 labelled tenant-B examples, a forward-pass budget, weight
inspection/edit tools, counterfactual probes, and the domain rules (はめあい公差 ⇒ リーマ穴; line
weight is meaningless). Held-out test sets for A and B — never shown to the agent — measure the fix
and regressions.

Compared:

``before``             the deployed model
``wsp_route``          agent's weight edit (scripted policy: reroute the existing tolerance detector
                       through a dead fusion unit into the ream/drill logits)
``wsp_route_ablate``   the same + mean-ablation of the line-weight input (domain rule 2 taken literally)
``ft_head``/``ft_all`` gradient fine-tuning of the classifier head / all weights on the same 16 samples

Run:  ``uv run wsp exp3 [--agent scripted|claude] [--seeds 0 1 2]``
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field

from wsp.agents.base import AgentResult, Task
from wsp.agents.prompts import EXP3_SYSTEM, exp3_user
from wsp.data.drawings import (
    CLASS_JA,
    CLASSES,
    FIT_TOLERANCES,
    GEO_FEATURES,
    HoleSample,
    Vocab,
    balanced_fewshot,
    collate,
    labels_of,
    make_probe,
    make_tenant_dataset,
)
from wsp.experiments.common import DEFAULT_RESULTS, env_info, make_agent, write_json
from wsp.harness.editor import WeightEditor, apply_patch
from wsp.harness.inspector import ForwardBudget, WeightInspector
from wsp.harness.sandbox import PythonSandbox, weights_from_module
from wsp.harness.tools import ToolInput, ToolRegistry, add_finish_tool, add_weight_tools
from wsp.models.drawing_head import DrawingHead, classification_report, train_drawing_head

FEWSHOT_PER_CLASS = 4
FORWARD_BUDGET = 3000

# ----------------------------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------------------------


class NoInput(ToolInput):
    pass


class ProbeItem(ToolInput):
    text: str = Field(description="Annotation text, e.g. 'φ10H7' or '2-φ8 キリ'.")
    like_fewshot: int | None = Field(None, description="Copy the geometry features of few-shot sample i.")
    geometry: dict[str, float] | None = Field(
        None, description=f"Geometry features by name (missing ones = 0): {list(GEO_FEATURES)}"
    )


class ProbeInput(ToolInput):
    items: list[ProbeItem] = Field(min_length=1, max_length=256)
    capture: list[str] = Field(
        default_factory=list,
        description="Module names whose outputs to return per item (e.g. 'fuse_act', 'text_enc').",
    )


class ActStatsInput(ToolInput):
    module: str = Field(description="Module name, e.g. 'fuse_act' or 'geo_enc.3'.")


class FixReport(ToolInput):
    diagnosis: str = Field(description="Root cause found in the weights, with evidence.")
    edit_description: str = Field(description="Which weights were changed and why.")
    expected_side_effects: str = Field(description="What could regress and how it was checked.")


def model_card(vocab: Vocab, n_fewshot: int) -> dict[str, Any]:
    return {
        "classes": [f"{i}: {c} ({CLASS_JA[c]})" for i, c in enumerate(CLASSES)],
        "geometry_features": list(GEO_FEATURES),
        "modules": {
            "geo_enc": "Sequential(Linear 8→32, ReLU, Linear 32→32, ReLU) on geometry features",
            "tok_emb": "EmbeddingBag(mean) over annotation tokens",
            "text_enc": "Sequential(Linear 32→32, ReLU)",
            "fuse": "Linear(64→64) on concat[geo_enc(32), text_enc(32)]",
            "fuse_act": "ReLU",
            "cls": "Linear(64→4) class logits",
            "aux_tol": "Linear(32→1) on text_enc: auxiliary 'has fit tolerance' logit (inspection planning)",
        },
        "tokenizer": "numbers → <NUM>, φd → <DIA>, Mx → <THREAD>, counts 'n-' → <N->; tolerance codes kept (e.g. H7)",
        "fit_tolerance_tokens_in_vocab": [t for t in FIT_TOLERANCES if t in vocab.stoi],
        "fewshot": f"{n_fewshot} labelled tenant-B samples (use evaluate_fewshot)",
        "forward_budget": FORWARD_BUDGET,
    }


class FixEnv:
    """Holds the model, few-shot data and budget for one fix session."""

    def __init__(self, model: DrawingHead, vocab: Vocab, fewshot: list[HoleSample]):
        self.model, self.vocab, self.fewshot = model, vocab, fewshot
        self.budget = ForwardBudget(FORWARD_BUDGET)
        self.inspector = WeightInspector(model, self.budget)
        self.editor = WeightEditor(model, max_rel_change=5.0)

    def _probs(
        self, samples: list[HoleSample], capture: list[str] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        b = collate(samples, self.vocab)
        acts = self.inspector.capture(b, [*(capture or []), "aux_tol", "cls"])
        logits = acts.pop("cls")
        aux = acts.pop("aux_tol").squeeze(-1)
        return logits.softmax(-1), aux, acts

    def evaluate_fewshot(self) -> dict[str, Any]:
        probs, aux, _ = self._probs(self.fewshot)
        pred = probs.argmax(-1)
        y = labels_of(self.fewshot)
        conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        for t, p in zip(y.tolist(), pred.tolist(), strict=True):
            conf[t, p] += 1
        return {
            "accuracy": float((pred == y).float().mean()),
            "confusion(rows=true, cols=pred)": conf.tolist(),
            "samples": [
                {
                    "i": i,
                    "text": s.text,
                    "label": CLASSES[s.label],
                    "pred": CLASSES[int(p)],
                    "p_pred": round(float(probs[i, p]), 3),
                    "aux_tol_logit": round(float(aux[i]), 3),
                    "geo": {k: round(float(v), 3) for k, v in zip(GEO_FEATURES, s.geo, strict=True)},
                }
                for i, (s, p) in enumerate(zip(self.fewshot, pred.tolist(), strict=True))
            ],
            "budget": self.budget.to_dict(),
        }

    def probe(self, a: ProbeInput) -> dict[str, Any]:
        samples = []
        for it in a.items:
            if it.like_fewshot is not None:
                geo = self.fewshot[it.like_fewshot].geo
            elif it.geometry is not None:
                unknown = set(it.geometry) - set(GEO_FEATURES)
                if unknown:
                    raise ValueError(f"unknown geometry features {sorted(unknown)}")
                geo = np.array([it.geometry.get(k, 0.0) for k in GEO_FEATURES], dtype=np.float32)
            else:
                raise ValueError("each probe item needs like_fewshot or geometry")
            samples.append(make_probe(geo, it.text))
        probs, aux, acts = self._probs(samples, a.capture)
        rows = []
        for i, it in enumerate(a.items):
            row = {
                "text": it.text,
                "pred": CLASSES[int(probs[i].argmax())],
                "probs": {c: round(float(probs[i, k]), 4) for k, c in enumerate(CLASSES)},
                "aux_tol_logit": round(float(aux[i]), 4),
            }
            for m, v in acts.items():
                row[m] = np.round(v[i].double().numpy(), 4).tolist()
            rows.append(row)
        return {"items": rows, "budget": self.budget.to_dict()}

    def activation_stats(self, a: ActStatsInput) -> dict[str, Any]:
        return {
            **self.inspector.activation_stats(collate(self.fewshot, self.vocab), [a.module]),
            "budget": self.budget.to_dict(),
        }

    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        add_weight_tools(
            reg,
            self.inspector,
            self.editor,
            sandbox_factory=lambda: PythonSandbox(weights_from_module(self.model), {"note": "DrawingHead weights"}),
        )
        reg.add(
            "evaluate_fewshot",
            "Accuracy, confusion and per-sample predictions on the labelled tenant-B "
            "few-shot set (costs one forward pass per sample).",
            NoInput,
            lambda _: self.evaluate_fewshot(),
        )
        reg.add(
            "probe",
            "Run hand-made inputs (e.g. counterfactual pairs with/without 'H7') through the model; "
            "optionally capture intermediate activations. Costs one forward pass per item.",
            ProbeInput,
            self.probe,
        )
        reg.add(
            "activation_stats",
            "Per-unit activation statistics of a module on the few-shot set (finds dead "
            "units). Costs one forward pass per few-shot sample.",
            ActStatsInput,
            self.activation_stats,
        )
        add_finish_tool(reg, FixReport, "Submit the fix with diagnosis and expected side effects.", name="submit_fix")
        return reg


# ----------------------------------------------------------------------------------------------
# Baselines: gradient fine-tuning on the same few-shot set
# ----------------------------------------------------------------------------------------------


def finetune(
    model: DrawingHead, vocab: Vocab, fewshot: list[HoleSample], head_only: bool, steps: int = 200, seed: int = 0
) -> tuple[DrawingHead, dict[str, Any]]:
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    m.train()
    params = list(m.cls.parameters()) if head_only else list(m.parameters())
    opt = torch.optim.Adam(params, lr=1e-2 if head_only else 1e-3)
    b, y = collate(fewshot, vocab), labels_of(fewshot)
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = F.cross_entropy(m(b["geo"], b["tokens"]), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    m.eval()
    changed = sum(int((p1 != p0).sum()) for p1, p0 in zip(m.parameters(), model.parameters(), strict=True))
    total = sum(p.numel() for p in model.parameters())
    return m, {"gradient_steps": steps, "seconds": time.perf_counter() - t0, "changed_fraction": changed / total}


# ----------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------


def evaluate(model: DrawingHead, vocab: Vocab, test_a: list[HoleSample], test_b: list[HoleSample]) -> dict[str, Any]:
    ra, rb = classification_report(model, test_a, vocab), classification_report(model, test_b, vocab)
    return {
        "A_accuracy": ra["accuracy"],
        "B_accuracy": rb["accuracy"],
        "B_ream_recall": rb["per_class"]["ream"]["recall"],
        "B_drill_precision": rb["per_class"]["drill"]["precision"],
        "A_ream_recall": ra["per_class"]["ream"]["recall"],
        "A_report": ra,
        "B_report": rb,
    }


def run_seed(seed: int, agent: Any, out_dir: Path) -> dict[str, Any]:
    from wsp.agents.scripted import ScriptedFixer

    train_a = make_tenant_dataset("A", 6000, seed=100 + seed)
    test_a = make_tenant_dataset("A", 2000, seed=200 + seed)
    test_b = make_tenant_dataset("B", 2000, seed=300 + seed)
    vocab = Vocab.build(train_a)
    base, train_info = train_drawing_head(train_a, vocab)
    fewshot = balanced_fewshot("B", FEWSHOT_PER_CLASS, seed=400 + seed)
    total_params = sum(p.numel() for p in base.parameters())
    row: dict[str, Any] = {"seed": seed, "train": train_info, "total_params": total_params, "methods": {}}
    row["methods"]["before"] = {"gradient_steps": 0, "changed_fraction": 0.0, **evaluate(base, vocab, test_a, test_b)}

    variants: list[tuple[str, Any]] = [("wsp_route", agent or ScriptedFixer())]
    if agent is None:
        variants.append(("wsp_route_ablate", ScriptedFixer(ablate_line_weight=True)))
    for name, ag in variants:
        model = copy.deepcopy(base)
        env = FixEnv(model, vocab, fewshot)
        reg = env.registry()
        task = Task(
            f"exp3_seed{seed}",
            EXP3_SYSTEM,
            exp3_user(model_card(vocab, len(fewshot))),
            context={
                "geo_features": GEO_FEATURES,
                "classes": CLASSES,
                "tolerance_tokens": [t for t in FIT_TOLERANCES if t in vocab.stoi],
            },
        )
        t0 = time.perf_counter()
        result: AgentResult = ag.run(task, reg)
        secs = time.perf_counter() - t0
        patch_path = out_dir / "patches" / f"seed{seed}_{name}"
        manifest = env.editor.export_patch(patch_path, metadata={"seed": seed, "method": name})
        # Sanity check: the exported sparse patch reproduces the edited model on a fresh copy.
        replica = copy.deepcopy(base)
        apply_patch(replica, patch_path)
        row["methods"][name] = {
            "gradient_steps": 0,
            "seconds": secs,
            "forward_passes": env.budget.used,
            "changed_params": int(sum(p["n_changed"] for p in manifest["params"])),
            "changed_fraction": env.editor.changed_fraction(),
            "edits": [r.to_dict() for r in env.editor.history],
            "patch": str(patch_path.relative_to(out_dir)) + ".{json,npz}",
            "agent": result.to_dict(),
            **evaluate(model, vocab, test_a, test_b),
        }

    for name, head_only in (("ft_head", True), ("ft_all", False)):
        m, info = finetune(base, vocab, fewshot, head_only, seed=seed)
        row["methods"][name] = {**info, "forward_passes": None, **evaluate(m, vocab, test_a, test_b)}
    return row


KEYS = ("A_accuracy", "B_accuracy", "B_ream_recall", "B_drill_precision", "A_ream_recall", "changed_fraction")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in rows[0]["methods"]:
        out[m] = {}
        for k in KEYS:
            vals = [r["methods"][m][k] for r in rows if r["methods"][m].get(k) is not None]
            if vals:
                out[m][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        out[m]["gradient_steps"] = float(np.mean([r["methods"][m]["gradient_steps"] for r in rows]))
    return out


def to_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    a = summary["aggregate"]

    def ms(m: str, k: str, pct: bool = True) -> str:
        v = a[m].get(k)
        if not v:
            return "–"
        return f"{v['mean']:.1%} ± {v['std']:.1%}" if pct else f"{v['mean']:.4%}"

    lines = [
        f"# Experiment 3 — Fix the tolerance-blind hole classifier (agent: `{summary['agent']}`)",
        "",
        f"Seeds: {summary['seeds']}. Test sets: 2,000 tenant-A and 2,000 tenant-B holes (held out). "
        f"Few-shot: {FEWSHOT_PER_CLASS * len(CLASSES)} tenant-B samples. Forward budget: {FORWARD_BUDGET}.",
        "",
        "| method | B accuracy | B リーマ recall | B キリ precision | A accuracy (regression) | A リーマ recall "
        "| changed weights | gradient steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in a:
        lines.append(
            f"| {m} | {ms(m, 'B_accuracy')} | {ms(m, 'B_ream_recall')} | {ms(m, 'B_drill_precision')} | "
            f"{ms(m, 'A_accuracy')} | {ms(m, 'A_ream_recall')} | {ms(m, 'changed_fraction', pct=False)} | "
            f"{a[m]['gradient_steps']:.0f} |"
        )
    lines += ["", "Per seed (wsp_route): forward passes used / edits / changed parameters", ""]
    for r in rows:
        w = r["methods"]["wsp_route"]
        lines.append(
            f"- seed {r['seed']}: {w['forward_passes']} forward passes, {len(w['edits'])} edits, "
            f"{w['changed_params']} / {r['total_params']} parameters changed, patch `{w['patch']}`"
        )
    return "\n".join(lines) + "\n"


def run(
    agent_name: str = "scripted", seeds: tuple[int, ...] = (0, 1, 2), out_dir: Path | None = None, **agent_kwargs: Any
) -> dict[str, Any]:
    out_dir = Path(out_dir or DEFAULT_RESULTS / "exp3")
    agent = make_agent(agent_name, **agent_kwargs)
    rows = []
    for seed in seeds:
        print(f"[exp3] seed {seed}", flush=True)
        row = run_seed(seed, agent, out_dir)
        rows.append(row)
        b, w = row["methods"]["before"], row["methods"]["wsp_route"]
        print(
            f"   B acc {b['B_accuracy']:.3f} → {w['B_accuracy']:.3f}, B ream recall {b['B_ream_recall']:.3f} → "
            f"{w['B_ream_recall']:.3f}, A acc {b['A_accuracy']:.3f} → {w['A_accuracy']:.3f}"
        )
    summary = {
        "experiment": "exp3_fix_tolerance_bias",
        "agent": agent_name if agent is None else agent.name,
        "seeds": list(seeds),
        "aggregate": aggregate(rows),
        "env": env_info(),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "cases.json", rows)
    (out_dir / "README.md").write_text(to_markdown(summary, rows))
    print(f"[exp3] done → {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="scripted", choices=["scripted", "claude"])
    p.add_argument("--model", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args(argv)
    kw = {"model": a.model} if a.model else {}
    run(a.agent, tuple(a.seeds), a.out_dir, **kw)


if __name__ == "__main__":
    main()
