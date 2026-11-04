"""Transactional, audited weight edits (the *Fix* / *Write* primitives for fixed architectures).

Design goals for edits made by an LLM agent in a customer environment:

* **Transactions with rollback** – ``with editor.transaction(validate=...)``: if the block raises
  or the validator rejects the result, every tensor is restored bit-for-bit.
* **Guard rails** – protected parameters, NaN/Inf rejection, optional cap on the relative change
  of any single edit.
* **Audit trail** – every edit is an :class:`EditRecord` (what, where, why, how much).
* **Portable hot-fix patches** – ``export_patch`` writes only the changed entries plus a SHA-256
  of the base weights, so the patch can be shipped into a closed network and applied only to
  the exact model it was made for (``apply_patch``).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from wsp.harness.inspector import parse_index


class EditRejected(RuntimeError):
    pass


@dataclass
class EditRecord:
    step: int
    op: str
    param: str
    index: str | None
    reason: str
    n_changed: int
    delta_l2: float
    rel_change: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


ValidateFn = Callable[[], "bool | tuple[bool, str]"]


class WeightEditor:
    def __init__(
        self,
        model: nn.Module,
        protected: Iterable[str] = (),
        max_rel_change: float | None = None,
    ):
        self.model = model
        self.protected = set(protected)
        self.max_rel_change = max_rel_change
        self.history: list[EditRecord] = []
        self._base = self.snapshot()
        self.base_hash = state_hash(model)
        self._checkpoints: list[tuple[str, dict[str, torch.Tensor], int]] = []

    # --- snapshots -------------------------------------------------------------------------------
    def snapshot(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def _restore(self, snap: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                v.copy_(snap[k])

    def checkpoint(self, label: str = "") -> int:
        self._checkpoints.append((label, self.snapshot(), len(self.history)))
        return len(self._checkpoints) - 1

    def rollback(self, to: int | None = None) -> dict[str, Any]:
        """Restore checkpoint ``to`` (default: the latest) and drop later checkpoints and edits."""
        if not self._checkpoints:
            # Rolling back with no checkpoint restores the original base weights.
            self._restore(self._base)
            dropped = len(self.history)
            self.history.clear()
            return {"restored": "base", "dropped_edits": dropped}
        to = len(self._checkpoints) - 1 if to is None else to
        label, snap, n_hist = self._checkpoints[to]
        self._restore(snap)
        dropped = len(self.history) - n_hist
        del self.history[n_hist:]
        del self._checkpoints[to:]
        return {"restored": label or f"checkpoint {to}", "dropped_edits": dropped}

    @contextmanager
    def transaction(self, label: str = "", validate: ValidateFn | None = None) -> Iterator[WeightEditor]:
        cp = self.checkpoint(label)
        try:
            yield self
            if validate is not None:
                verdict = validate()
                ok, msg = verdict if isinstance(verdict, tuple) else (bool(verdict), "")
                if not ok:
                    raise EditRejected(f"validation failed for transaction {label!r}: {msg}")
        except BaseException:
            self.rollback(cp)
            raise
        else:
            del self._checkpoints[cp:]  # commit: keep the weights, discard the checkpoint

    # --- edits -----------------------------------------------------------------------------------
    def _param(self, name: str) -> torch.nn.Parameter:
        params = dict(self.model.named_parameters())
        if name not in params:
            raise KeyError(f"unknown parameter {name!r}")
        if name in self.protected:
            raise EditRejected(f"parameter {name!r} is protected")
        return params[name]

    def _apply(
        self, op: str, name: str, index: str | None, fn: Callable[[torch.Tensor], torch.Tensor], reason: str
    ) -> EditRecord:
        p = self._param(name)
        idx = parse_index(index)
        with torch.no_grad():
            old = p.data[idx].clone()
            new = fn(old).to(p.dtype)
            if new.shape != old.shape:
                new = new.expand_as(old).clone()
            if not torch.isfinite(new).all():
                raise EditRejected(f"edit on {name}[{index}] produces non-finite values")
            delta = (new - old).double()
            rel = float(delta.norm() / (p.data.double().norm() + 1e-12))
            if self.max_rel_change is not None and rel > self.max_rel_change:
                raise EditRejected(
                    f"edit on {name}[{index}] changes the tensor by {rel:.3f} (relative) > cap {self.max_rel_change}"
                )
            p.data[idx] = new
        rec = EditRecord(
            step=len(self.history),
            op=op,
            param=name,
            index=index,
            reason=reason,
            n_changed=int((delta != 0).sum()),
            delta_l2=float(delta.norm()),
            rel_change=rel,
            timestamp=time.time(),
        )
        self.history.append(rec)
        return rec

    def set(self, name: str, index: str | None, value: Any, reason: str = "") -> EditRecord:
        v = torch.as_tensor(np.asarray(value, dtype=np.float64))
        return self._apply("set", name, index, lambda old: v.to(old.dtype).expand_as(old).clone(), reason)

    def add(self, name: str, index: str | None, delta: Any, reason: str = "") -> EditRecord:
        d = torch.as_tensor(np.asarray(delta, dtype=np.float64))
        return self._apply("add", name, index, lambda old: old + d.to(old.dtype), reason)

    def scale(self, name: str, index: str | None, factor: float, reason: str = "") -> EditRecord:
        return self._apply("scale", name, index, lambda old: old * float(factor), reason)

    # --- reporting / patches ---------------------------------------------------------------------
    def diff(self) -> dict[str, dict[str, Any]]:
        out = {}
        for k, v in self.model.state_dict().items():
            d = (v.detach() - self._base[k]).double()
            n = int((d != 0).sum())
            if n:
                out[k] = {
                    "n_changed": n,
                    "numel": v.numel(),
                    "delta_l2": float(d.norm()),
                    "rel_change": float(d.norm() / (self._base[k].double().norm() + 1e-12)),
                }
        return out

    def changed_fraction(self) -> float:
        total = sum(v.numel() for v in self._base.values())
        changed = sum(d["n_changed"] for d in self.diff().values())
        return changed / total if total else 0.0

    def export_patch(self, path: str | Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write a sparse patch (``.npz``) and its manifest (``.json``) next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        params = []
        for k, v in self.model.state_dict().items():
            d = v.detach() - self._base[k]
            flat_idx = torch.nonzero(d.flatten() != 0).flatten()
            if len(flat_idx):
                arrays[f"{k}::idx"] = flat_idx.numpy().astype(np.int64)
                arrays[f"{k}::val"] = v.detach().flatten()[flat_idx].numpy()
                params.append({"name": k, "shape": list(v.shape), "n_changed": len(flat_idx)})
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        manifest = {
            "format": "wsp-patch/1",
            "base_sha256": self.base_hash,
            "patched_sha256": state_hash(self.model),
            "params": params,
            "changed_fraction": self.changed_fraction(),
            "history": [r.to_dict() for r in self.history],
            "metadata": metadata or {},
        }
        path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest


def apply_patch(model: nn.Module, path: str | Path, verify: bool = True) -> dict[str, Any]:
    """Apply a patch written by :meth:`WeightEditor.export_patch` (checks the base hash)."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if verify and state_hash(model) != manifest["base_sha256"]:
        raise EditRejected("patch base hash does not match this model — refusing to apply")
    data = np.load(path.with_suffix(".npz"))
    sd = model.state_dict()
    with torch.no_grad():
        for p in manifest["params"]:
            t = sd[p["name"]]
            flat = t.view(-1)
            flat[torch.as_tensor(data[f"{p['name']}::idx"])] = torch.as_tensor(data[f"{p['name']}::val"]).to(t.dtype)
    if verify and state_hash(model) != manifest["patched_sha256"]:
        raise EditRejected("patched model hash mismatch after applying patch")
    return manifest
