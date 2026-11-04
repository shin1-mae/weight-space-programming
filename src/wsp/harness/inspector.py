"""Read-only access to a model's weights and activations (the *Read* primitive).

Everything returned is JSON-serialisable and size-capped so it can be handed to an LLM as a
tool result. Forward passes are metered by a :class:`ForwardBudget`, mirroring the limited
"forward-pass budget" setting used for weight-space debugging.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ForwardBudget:
    """Counts per-example forward passes. ``limit=None`` means unlimited."""

    limit: int | None = None
    used: int = 0

    def consume(self, n: int) -> None:
        if self.limit is not None and self.used + n > self.limit:
            raise BudgetExceeded(f"forward budget exceeded: used {self.used} + {n} > limit {self.limit}")
        self.used += n

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else self.limit - self.used

    def to_dict(self) -> dict[str, int | None]:
        return {"limit": self.limit, "used": self.used, "remaining": self.remaining}


_SLICE_RE = re.compile(r"^\s*(-?\d+)?\s*:\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*)?$")


def parse_index(spec: str | None) -> tuple[Any, ...]:
    """Parse a numpy-style index string such as ``"3, :"`` or ``"0:8, 2"`` without ``eval``."""
    if spec is None or spec.strip() in ("", ":", "..."):
        return (slice(None),)
    out: list[Any] = []
    for part in spec.split(","):
        part = part.strip()
        if re.fullmatch(r"-?\d+", part):
            out.append(int(part))
            continue
        m = _SLICE_RE.match(part)
        if not m:
            raise ValueError(f"unsupported index component: {part!r}")
        a, b, c = (int(g) if g not in (None, "") else None for g in m.groups())
        out.append(slice(a, b, c))
    return tuple(out)


def _r(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


class WeightInspector:
    """Statistics, spectra, slices and activation probes over an ``nn.Module``."""

    def __init__(
        self,
        model: nn.Module,
        budget: ForwardBudget | None = None,
        call: Callable[[nn.Module, Any], torch.Tensor] | None = None,
    ):
        self.model = model
        self.budget = budget or ForwardBudget()
        self._call = call or (lambda m, batch: m(**batch) if isinstance(batch, dict) else m(batch))

    # --- static weight reading -------------------------------------------------------------------
    def _param(self, name: str) -> torch.Tensor:
        params = dict(self.model.named_parameters())
        if name not in params:
            raise KeyError(f"unknown parameter {name!r}; available: {sorted(params)}")
        return params[name].detach()

    def list_parameters(self) -> list[dict[str, Any]]:
        return [
            {"name": n, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype).replace("torch.", "")}
            for n, p in self.model.named_parameters()
        ]

    def tensor_stats(self, name: str) -> dict[str, Any]:
        t = self._param(name).double()
        flat = t.flatten()
        amax = float(flat.abs().max()) if flat.numel() else 0.0
        out: dict[str, Any] = {
            "name": name,
            "shape": list(t.shape),
            "mean": _r(flat.mean()),
            "std": _r(flat.std()) if flat.numel() > 1 else 0.0,
            "min": _r(flat.min()),
            "max": _r(flat.max()),
            "l2": _r(flat.norm()),
            "abs_mean": _r(flat.abs().mean()),
            "frac_near_zero": _r((flat.abs() < 1e-3 * max(amax, 1e-12)).double().mean()),
        }
        if t.ndim == 2:
            rn, cn = t.norm(dim=1), t.norm(dim=0)
            out["row_norms"] = {"min": _r(rn.min()), "mean": _r(rn.mean()), "max": _r(rn.max())}
            out["col_norms"] = {"min": _r(cn.min()), "mean": _r(cn.mean()), "max": _r(cn.max())}
            out["argmax_col_norm"] = int(cn.argmax())
        return out

    def svd(self, name: str, k: int = 8) -> dict[str, Any]:
        t = self._param(name).double()
        if t.ndim != 2:
            raise ValueError(f"{name} is not a matrix (shape {list(t.shape)})")
        s = torch.linalg.svdvals(t)
        p = s / s.sum()
        entropy = float(-(p[p > 0] * p[p > 0].log()).sum())
        return {
            "name": name,
            "singular_values": [_r(v, 5) for v in s[:k]],
            "stable_rank": _r((s**2).sum() / s[0] ** 2, 4),
            "effective_rank": _r(np.exp(entropy), 4),
        }

    def read_slice(self, name: str, index: str | None = None, max_elements: int = 512) -> dict[str, Any]:
        t = self._param(name)
        sub = t[parse_index(index)]
        if sub.numel() > max_elements:
            raise ValueError(f"slice has {sub.numel()} elements > max_elements={max_elements}; narrow the index")
        return {
            "name": name,
            "index": index,
            "shape": list(sub.shape),
            "values": np.round(sub.double().numpy(), 6).tolist(),
        }

    def top_k(self, name: str, k: int = 10, by_abs: bool = True) -> list[dict[str, Any]]:
        t = self._param(name).double()
        key = t.abs() if by_abs else t
        vals, idx = key.flatten().topk(min(k, key.numel()))
        coords = np.unravel_index(idx.numpy(), tuple(t.shape))
        return [{"index": [int(c[i]) for c in coords], "value": _r(t.flatten()[idx[i]])} for i in range(len(vals))]

    # --- dynamic probing (metered) ---------------------------------------------------------------
    def _batch_size(self, batch: Any) -> int:
        if isinstance(batch, dict):
            return int(next(iter(batch.values())).shape[0])
        return int(batch.shape[0])

    @torch.no_grad()
    def forward(self, batch: Any) -> torch.Tensor:
        self.budget.consume(self._batch_size(batch))
        return self._call(self.model, batch)

    @torch.no_grad()
    def capture(self, batch: Any, modules: Iterable[str]) -> dict[str, torch.Tensor]:
        """Run one metered forward pass and return the outputs of the named sub-modules."""
        named = dict(self.model.named_modules())
        acts: dict[str, torch.Tensor] = {}
        hooks = []
        for m in modules:
            if m not in named:
                raise KeyError(f"unknown module {m!r}; available: {[n for n in named if n]}")
            hooks.append(named[m].register_forward_hook(lambda _mod, _inp, out, m=m: acts.__setitem__(m, out.detach())))
        try:
            self.forward(batch)
        finally:
            for h in hooks:
                h.remove()
        return acts

    def activation_stats(self, batch: Any, modules: Iterable[str]) -> dict[str, dict[str, Any]]:
        acts = self.capture(batch, modules)
        out = {}
        for m, a in acts.items():
            a2 = a.reshape(a.shape[0], -1).double()
            out[m] = {
                "units": a2.shape[1],
                "mean": np.round(a2.mean(0).numpy(), 5).tolist(),
                "frac_active": np.round((a2 > 0).double().mean(0).numpy(), 4).tolist(),
                "dead_units": [int(i) for i in torch.nonzero((a2 > 0).sum(0) == 0).flatten()],
            }
        return out
