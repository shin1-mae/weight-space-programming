"""Shared experiment plumbing: output folders, JSON writing, agent selection, model caches."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wsp.geometry.parts import MM_PER_UNIT, PartSpec, make_part_set, part_sdf_norm
from wsp.models.cad_sdf_mlp import SDFMLP, FitInfo, SDFTrainConfig, fit_sdf

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def env_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def make_agent(name: str, **kwargs: Any):
    """``scripted`` (offline reference policy) or ``claude`` (Claude via the Anthropic API)."""
    if name == "claude":
        from wsp.agents.llm import ClaudeAgent

        return ClaudeAgent(**kwargs)
    if name != "scripted":
        raise ValueError(f"unknown agent {name!r} (use 'scripted' or 'claude')")
    return None  # each experiment instantiates its own scripted policy


# ----------------------------------------------------------------------------------------------
# "Customer" SDF models shared by experiments 1 and 2
# ----------------------------------------------------------------------------------------------


@dataclass
class PartModel:
    spec: PartSpec
    model: SDFMLP
    fit: FitInfo


def spec_bbox_norm(spec: PartSpec) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = spec.bbox_mm()
    return lo / MM_PER_UNIT, hi / MM_PER_UNIT


def load_or_fit_parts(
    n_per_family: int = 2,
    seed: int = 0,
    cache_dir: Path | None = None,
    cfg: SDFTrainConfig | None = None,
    log: bool = True,
) -> list[PartModel]:
    """Fit (or load from ``cache_dir``) one SDF network per generated part."""
    cfg = cfg or SDFTrainConfig()
    cache_dir = cache_dir or DEFAULT_ARTIFACTS / "sdf_models" / f"seed{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for spec in make_part_set(n_per_family, seed=seed):
        path = cache_dir / f"{spec.name}.pt"
        meta_path = cache_dir / f"{spec.name}.json"
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("spec") == _jsonified(spec.to_dict()) and meta.get("train_cfg") == _jsonified(cfg.to_dict()):
                model = SDFMLP(cfg.hidden)
                model.load_state_dict(torch.load(path, weights_only=True))
                model.eval()
                out.append(PartModel(spec, model, FitInfo(**meta["fit"])))
                continue
        if log:
            print(f"  fitting SDF network for {spec.name} ...", flush=True)
        model, fit = fit_sdf(lambda p, s=spec: part_sdf_norm(s, p), spec_bbox_norm(spec), cfg)
        torch.save(model.state_dict(), path)
        write_json(meta_path, {"spec": spec.to_dict(), "train_cfg": cfg.to_dict(), "fit": fit.__dict__})
        out.append(PartModel(spec, model, fit))
    return out


def _jsonified(d: dict[str, Any]) -> Any:
    """What ``d`` looks like after a JSON round trip (tuples become lists)."""
    return json.loads(json.dumps(d, default=_json_default))


def fmt(x: float | None, nd: int = 2) -> str:
    if x is None:
        return "–"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "n/a"
    return f"{x:.{nd}f}"
