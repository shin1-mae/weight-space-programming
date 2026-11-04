"""Implicit neural representation of a 3D machine part: a small ReLU MLP that maps a point
``(x, y, z)`` (normalised, 1 unit = 50 mm) to its signed distance (normalised units).

The network is deliberately plain — ``Linear → ReLU → … → Linear`` — so that its weights can be
read and rewritten as an explicit program: the first layer is a set of oriented planes in 3D,
deeper layers combine them piecewise-linearly, and the last layer is a weighted sum.

``to_layers`` / ``from_layers`` convert between the torch module and a plain list of
``(W, b)`` numpy arrays ("the weight program") that ``wsp.geometry.sdf_ops`` rewrites.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from wsp.geometry.mesh import extract_mesh, sample_surface

Layers = list[tuple[np.ndarray, np.ndarray]]


class SDFMLP(nn.Module):
    """ReLU MLP ``R^3 -> R``. Parameters are named ``net.{2i}.weight`` / ``net.{2i}.bias``."""

    def __init__(self, hidden: Sequence[int] = (64, 64, 64), in_dim: int = 3):
        super().__init__()
        dims = [in_dim, *hidden, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    @property
    def hidden_sizes(self) -> list[int]:
        return [m.out_features for m in self.net if isinstance(m, nn.Linear)][:-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    # --- weight-program conversion -------------------------------------------------------------
    def to_layers(self) -> Layers:
        return [
            (m.weight.detach().cpu().double().numpy().copy(), m.bias.detach().cpu().double().numpy().copy())
            for m in self.net
            if isinstance(m, nn.Linear)
        ]

    @classmethod
    def from_layers(cls, layers: Layers) -> SDFMLP:
        hidden = [w.shape[0] for w, _ in layers[:-1]]
        model = cls(hidden=hidden, in_dim=layers[0][0].shape[1])
        linears = [m for m in model.net if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for lin, (w, b) in zip(linears, layers, strict=True):
                lin.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
                lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
        return model

    @torch.no_grad()
    def sdf_numpy(self, pts: np.ndarray, chunk: int = 262_144) -> np.ndarray:
        out = []
        for i in range(0, len(pts), chunk):
            out.append(self(torch.as_tensor(pts[i : i + chunk], dtype=torch.float32)).numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def mlp_forward_np(layers: Layers, x: np.ndarray) -> np.ndarray:
    """Reference numpy forward pass of a ReLU MLP weight program (float64)."""
    h = np.asarray(x, dtype=np.float64)
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h.squeeze(-1)


# ----------------------------------------------------------------------------------------------
# Fitting (the *only* place where gradient descent is used for SDFs — to create the
# pre-trained "customer model" and for the retraining baseline in Experiment 2).
# ----------------------------------------------------------------------------------------------


@dataclass
class SDFTrainConfig:
    hidden: tuple[int, ...] = (64, 64, 64)
    steps: int = 3000
    batch_size: int = 4096
    lr: float = 2e-3
    clamp: float = 0.1  # normalised units (= 5 mm); DeepSDF-style clamped L1
    n_surface: int = 300_000
    n_uniform: int = 100_000
    surface_sigmas: tuple[float, ...] = (0.004, 0.02)  # normalised units (0.2 mm, 1 mm)
    mesh_resolution: float = 0.012  # normalised units (0.6 mm) for surface sampling
    feature_weight: float = 16.0  # oversample hole walls, contours and inner floors (see sample_surface)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FitInfo:
    final_loss: float
    seconds: float
    steps: int
    gradient_steps: int


def make_training_set(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    bbox: tuple[np.ndarray, np.ndarray],
    cfg: SDFTrainConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
    pad = 0.06
    mesh = extract_mesh(sdf_fn, lo - pad, hi + pad, cfg.mesh_resolution)
    surf = sample_surface(mesh, cfg.n_surface, rng, feature_weight=cfg.feature_weight)
    sig = rng.choice(np.asarray(cfg.surface_sigmas), size=(len(surf), 1))
    near = surf + rng.normal(size=surf.shape) * sig
    # Uniform samples: half over the whole workspace, half over the padded bounding box.
    n_box = cfg.n_uniform // 2
    uni = np.concatenate(
        [
            rng.uniform(-1, 1, size=(cfg.n_uniform - n_box, 3)),
            rng.uniform(lo - 4 * pad, hi + 4 * pad, size=(n_box, 3)),
        ]
    )
    pts = np.concatenate([near, uni]).clip(-1, 1)
    return pts.astype(np.float32), np.asarray(sdf_fn(pts), dtype=np.float32)


def fit_sdf(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    bbox: tuple[np.ndarray, np.ndarray],
    cfg: SDFTrainConfig | None = None,
) -> tuple[SDFMLP, FitInfo]:
    """Fit an ``SDFMLP`` to an analytic SDF (normalised units) with clamped L1 regression."""
    cfg = cfg or SDFTrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    pts, sdf = make_training_set(sdf_fn, bbox, cfg, rng)
    x_all = torch.from_numpy(pts)
    y_all = torch.from_numpy(sdf).clamp(-cfg.clamp, cfg.clamp)

    model = SDFMLP(cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps, eta_min=cfg.lr * 0.02)
    gen = torch.Generator().manual_seed(cfg.seed)
    t0 = time.perf_counter()
    loss_val = float("nan")
    for _ in range(cfg.steps):
        idx = torch.randint(0, len(x_all), (cfg.batch_size,), generator=gen)
        pred = model(x_all[idx]).clamp(-cfg.clamp, cfg.clamp)
        loss = (pred - y_all[idx]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_val = float(loss.detach())
    model.eval()
    info = FitInfo(final_loss=loss_val, seconds=time.perf_counter() - t0, steps=cfg.steps, gradient_steps=cfg.steps)
    return model, info
