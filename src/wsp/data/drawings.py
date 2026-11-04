"""Synthetic hole-callout dataset for 2D drawings (図面の穴指示), with two customer "tenants".

Each sample is one hole feature extracted from a drawing:

* **geometry features** (from the view/section geometry): diameter, depth ratio, through flag,
  counterbore step, thread hidden-lines, pattern count — and the *pen line weight* of the hole
  circle, which depends only on the customer's CAD template;
* **annotation text** (寸法・注記): e.g. ``"4-φ10H7 通し"``, ``"M8 深さ12"``,
  ``"φ9 ザグリφ14 深さ8.5"``.

Classes: キリ穴 (drill), リーマ穴 (ream: tight fit tolerance such as H7), タップ穴 (tap),
ザグリ穴 (counterbore). A drilled and a reamed hole look *identical* in the geometry; only the
fit-tolerance call-out (H7/H6/G7 …) tells them apart.

Tenant A (training data) draws reamed holes with a thick pen (0.5 mm) and puts the tolerance in a
separate tolerance table for most of them ("公差表参照"), so a model trained on A learns the
shortcut *thick line ⇒ reamed*. Tenant B (deployment) uses a uniform thin pen and always writes the
tolerance next to the diameter — the shortcut fails and reamed holes are classified as drilled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

CLASSES: tuple[str, ...] = ("drill", "ream", "tap", "counterbore")
CLASS_JA = {"drill": "キリ穴", "ream": "リーマ穴", "tap": "タップ穴", "counterbore": "ザグリ穴"}
GEO_FEATURES: tuple[str, ...] = (
    "diameter_mm/20",
    "depth_ratio",
    "is_through",
    "has_step",
    "step_ratio/2",
    "thread_hidden_lines",
    "line_weight_mm*2",
    "n_instances/8",
)
FIT_TOLERANCES = ("H6", "H7", "H8", "G6", "G7", "F7", "JS7")
Tenant = Literal["A", "B"]

TOKEN_RE = re.compile(
    r"(?P<count>\d+-)|(?P<dia>[φΦø]\s*\d+(?:\.\d+)?)|(?P<thread>M\d+(?:x[\d.]+)?)"
    r"|(?P<tol>[A-Z]{1,2}\d{1,2})|(?P<num>\d+(?:\.\d+)?)|(?P<word>[^\s\dφΦø]+)"
)


def tokenize(text: str) -> list[str]:
    """Normalise a call-out into tokens: numbers become placeholders, tolerances stay verbatim."""
    out: list[str] = []
    for m in TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == "count":
            out.append("<N->")
        elif kind == "dia":
            out.append("<DIA>")
        elif kind == "thread":
            out.append("<THREAD>")
        elif kind == "tol":
            out.append(m.group())
        elif kind == "num":
            out.append("<NUM>")
        else:
            out.append(m.group())
    return out


def has_fit_tolerance(tokens: list[str]) -> bool:
    return any(t in FIT_TOLERANCES for t in tokens)


@dataclass
class HoleSample:
    label: int
    geo: np.ndarray
    text: str
    tokens: list[str]


def _fmt(x: float) -> str:
    return f"{x:g}"


def sample_hole(cls: str, tenant: Tenant, rng: np.random.Generator) -> HoleSample:
    n = int(rng.choice([1, 1, 1, 2, 4, 6]))
    prefix = f"{n}-" if n > 1 else ""
    thickness = rng.uniform(8, 30)
    d = float(np.round(rng.uniform(3, 20) * 2) / 2)
    through, depth = True, thickness
    step, step_ratio, thread = 0.0, 0.0, 0.0
    # Pen weight comes from the tenant's CAD template: A draws reamed holes thick, B is uniformly thin.
    line_w = (0.5 if cls == "ream" else 0.25) if tenant == "A" else float(rng.choice([0.18, 0.25]))

    if cls == "drill":
        text = f"{prefix}φ{_fmt(d)}" + str(rng.choice(["キリ", " キリ", " 通し", " THRU", ""]))
    elif cls == "ream":
        tol = str(rng.choice(["H7"] * 6 + ["H6", "G7", "H8", "G6"]))
        explicit = rng.random() < (0.35 if tenant == "A" else 1.0)
        if explicit:
            text = f"{prefix}φ{_fmt(d)}{tol}" + str(rng.choice(["", " 通し", " THRU"]))
        else:
            text = f"{prefix}φ{_fmt(d)}" + str(rng.choice([" 公差表参照", " (※1)", "", " 通し"]))
    elif cls == "tap":
        m = int(rng.choice([3, 4, 5, 6, 8, 10, 12, 16]))
        d = m * 0.85
        thread = 1.0
        if rng.random() < 0.6:
            through, depth = False, float(np.round(rng.uniform(1.5, 2.5) * m))
            text = f"{prefix}M{m} 深さ{_fmt(depth)}"
        else:
            text = f"{prefix}M{m}" + str(rng.choice(["", " 通し", " THRU"]))
    elif cls == "counterbore":
        big = float(np.round(d * rng.uniform(1.5, 1.9) * 2) / 2)
        h = float(np.round(rng.uniform(0.6, 1.0) * d * 2) / 2)
        step, step_ratio = 1.0, big / d
        word = str(rng.choice(["ザグリ", "座ぐり", "CB"]))
        text = f"{prefix}φ{_fmt(d)} {word}φ{_fmt(big)} 深さ{_fmt(h)}"
    else:  # pragma: no cover
        raise ValueError(cls)

    geo = np.array(
        [
            d / 20.0,
            (depth / thickness) if not through else 1.0,
            1.0 if through else 0.0,
            step,
            step_ratio / 2.0,
            thread,
            line_w * 2.0,
            n / 8.0,
        ],
        dtype=np.float32,
    )
    geo = geo + rng.normal(0, 0.01, size=geo.shape).astype(np.float32)  # extraction noise
    return HoleSample(CLASSES.index(cls), geo, text, tokenize(text))


def make_tenant_dataset(
    tenant: Tenant, n: int, seed: int, class_probs: tuple[float, ...] | None = None
) -> list[HoleSample]:
    rng = np.random.default_rng(seed)
    probs = np.asarray(class_probs or (0.4, 0.25, 0.2, 0.15))
    labels = rng.choice(len(CLASSES), size=n, p=probs / probs.sum())
    return [sample_hole(CLASSES[c], tenant, rng) for c in labels]


def balanced_fewshot(tenant: Tenant, per_class: int, seed: int) -> list[HoleSample]:
    rng = np.random.default_rng(seed)
    return [sample_hole(c, tenant, rng) for c in CLASSES for _ in range(per_class)]


class Vocab:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, tokens: list[str]):
        self.itos = [self.PAD, self.UNK, *sorted(set(tokens) - {self.PAD, self.UNK})]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, samples: list[HoleSample], extra: tuple[str, ...] = FIT_TOLERANCES) -> Vocab:
        return cls([t for s in samples for t in s.tokens] + list(extra))

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, 1) for t in tokens] or [1]

    def __len__(self) -> int:
        return len(self.itos)


def collate(samples: list[HoleSample], vocab: Vocab) -> dict[str, torch.Tensor]:
    ids = [vocab.encode(s.tokens) for s in samples]
    width = max(len(i) for i in ids)
    tok = torch.zeros((len(samples), width), dtype=torch.long)
    for r, row in enumerate(ids):
        tok[r, : len(row)] = torch.as_tensor(row)
    return {"geo": torch.as_tensor(np.stack([s.geo for s in samples])), "tokens": tok}


def labels_of(samples: list[HoleSample]) -> torch.Tensor:
    return torch.as_tensor([s.label for s in samples], dtype=torch.long)


def aux_labels_of(samples: list[HoleSample]) -> torch.Tensor:
    return torch.as_tensor([float(has_fit_tolerance(s.tokens)) for s in samples])


def make_probe(geo: np.ndarray, text: str, label: int = 0) -> HoleSample:
    """Hand-made sample for counterfactual probing (e.g. the same hole with/without "H7")."""
    toks = tokenize(text)
    return HoleSample(label, np.asarray(geo, dtype=np.float32), text, toks)
