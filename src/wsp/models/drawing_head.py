"""Multimodal (geometry + call-out text) hole-type classifier for drawings.

::

    geo (8) ─► geo_enc: Linear→ReLU→Linear→ReLU ─► g (32) ─┐
                                                          ├─► fuse: Linear(64→64) ─► fuse_act: ReLU ─► cls (4)
    tokens ─► tok_emb: EmbeddingBag(mean) ─► text_enc ─► t (32) ─┘
                                                   └──────────► aux_tol: Linear(32→1)  "has fit tolerance?"

``aux_tol`` is an auxiliary head trained to detect fit-tolerance call-outs (H7, G6, …) — a
realistic secondary output used for inspection planning. It means the *capability* to recognise
tolerances exists inside ``text_enc``; whether the class head *uses* it is another matter.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from wsp.data.drawings import CLASSES, GEO_FEATURES, HoleSample, Vocab, aux_labels_of, collate, labels_of


class DrawingHead(nn.Module):
    def __init__(self, vocab_size: int, n_geo: int = len(GEO_FEATURES), d: int = 32, fuse_hidden: int = 64):
        super().__init__()
        self.geo_enc = nn.Sequential(nn.Linear(n_geo, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.tok_emb = nn.EmbeddingBag(vocab_size, d, mode="mean", padding_idx=0)
        self.text_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.fuse = nn.Linear(2 * d, fuse_hidden)
        self.fuse_act = nn.ReLU()
        self.cls = nn.Linear(fuse_hidden, len(CLASSES))
        self.aux_tol = nn.Linear(d, 1)
        self.d = d

    def encode(self, geo: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.geo_enc(geo), self.text_enc(self.tok_emb(tokens))

    def forward_all(self, geo: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, t = self.encode(geo, tokens)
        h = self.fuse_act(self.fuse(torch.cat([g, t], dim=-1)))
        return self.cls(h), self.aux_tol(t).squeeze(-1)

    def forward(self, geo: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_all(geo, tokens)[0]


@dataclass
class DrawingTrainConfig:
    epochs: int = 40
    batch_size: int = 128
    lr: float = 3e-3
    aux_weight: float = 0.5
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def train_drawing_head(
    train: list[HoleSample], vocab: Vocab, cfg: DrawingTrainConfig | None = None
) -> tuple[DrawingHead, dict]:
    cfg = cfg or DrawingTrainConfig()
    torch.manual_seed(cfg.seed)
    model = DrawingHead(len(vocab))
    batch = collate(train, vocab)
    y, y_aux = labels_of(train), aux_labels_of(train)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    t0 = time.perf_counter()
    n = len(train)
    steps = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            logits, aux = model.forward_all(batch["geo"][idx], batch["tokens"][idx])
            loss = F.cross_entropy(logits, y[idx]) + cfg.aux_weight * F.binary_cross_entropy_with_logits(
                aux, y_aux[idx]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
    model.eval()
    return model, {"seconds": time.perf_counter() - t0, "gradient_steps": steps, "final_loss": float(loss.detach())}


@torch.no_grad()
def predict(model: DrawingHead, samples: list[HoleSample], vocab: Vocab) -> torch.Tensor:
    b = collate(samples, vocab)
    return model(b["geo"], b["tokens"]).argmax(-1)


def classification_report(model: DrawingHead, samples: list[HoleSample], vocab: Vocab) -> dict:
    pred = predict(model, samples, vocab)
    y = labels_of(samples)
    per_class = {}
    for k, name in enumerate(CLASSES):
        mask = y == k
        tp = int(((pred == k) & mask).sum())
        per_class[name] = {
            "support": int(mask.sum()),
            "recall": tp / max(int(mask.sum()), 1),
            "precision": tp / max(int((pred == k).sum()), 1),
        }
    conf = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for t, p in zip(y.tolist(), pred.tolist(), strict=True):
        conf[t, p] += 1
    return {
        "accuracy": float((pred == y).float().mean()),
        "per_class": per_class,
        "confusion": conf.tolist(),
        "n": len(samples),
    }
