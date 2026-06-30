"""
VERSION A  —  FAST baseline  (~15 min on 6 CPU)

CLIP ViT-B/16 is FROZEN. We embed every image once, cache the vectors, then
train only a small Deep-SAD head on the cached vectors. No encoder backprop ->
training is trivial on CPU.

Trade-off: because the features are frozen, version A is WEAK on the hard
selfie+KTP case (the KTP dominates the CLIP embedding and selfie+KTP lands near
the positive manifold). Use this as the quick baseline; use train_b.py for the
real selfie+KTP attack.

Run:
    python -m clip_sad.train_a
Output:
    models/clip_sad_A.pt   (load with predict.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from clip_sad.sad_common import (
    SADConfig, build_index, load_clip, embed_paths,
    ProjectionHead, init_center, deep_sad_loss,
    scores_from_z, calibrate_threshold, save_bundle,
)

_PKG = Path(__file__).resolve().parent
OUT_PATH = str(_PKG / "models" / "clip_sad_A.pt")
CACHE_PATH = str(_PKG / "cache" / "clip_features_A.pt")


def main():
    cfg = SADConfig()
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    paths, labels, weights = build_index(cfg)
    print(f"dataset: {int((labels==1).sum())} pos / {int((labels==-1).sum())} neg "
          f"({len(paths)} total)")

    # ── 1. cache CLIP features once (the only forward pass over the encoder) ──
    cache = _load_or_build_cache(cfg, paths)
    feats = cache["feats"]                       # [N, 512] L2-normalized, frozen
    labels_t = torch.from_numpy(labels)
    weights_t = torch.from_numpy(weights)

    # ── 2. init head + center ─────────────────────────────────────────────────
    head = ProjectionHead(feats.shape[1], cfg.proj_dim)
    with torch.no_grad():
        z0 = head(feats)
    center = init_center(z0)

    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ds = TensorDataset(feats, labels_t, weights_t)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # ── 3. train head with Deep SAD loss ─────────────────────────────────────
    for epoch in range(cfg.epochs):
        head.train()
        running = 0.0
        for fb, lb, wb in dl:
            opt.zero_grad()
            z = head(fb)
            loss = deep_sad_loss(z, lb, wb, center, cfg.eta, cfg.eps)
            loss.backward()
            opt.step()
            running += loss.item() * len(fb)
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:3d}  loss {running/len(ds):.4f}")

    # ── 4. calibrate threshold on the full set ───────────────────────────────
    head.eval()
    with torch.no_grad():
        z_all = head(feats)
    scores = scores_from_z(z_all, center)
    threshold = calibrate_threshold(scores, labels, cfg.target_pos_recall)
    _report(scores, labels, threshold)

    save_bundle(OUT_PATH, mode="A", cfg=cfg, center=center,
                threshold=threshold, head=head, visual_state=None)
    print(f"saved -> {OUT_PATH}")


def _load_or_build_cache(cfg: SADConfig, paths: list[str]) -> dict:
    if Path(CACHE_PATH).exists():
        cache = torch.load(CACHE_PATH, map_location="cpu")
        if cache.get("paths") == paths:
            print(f"using cached features: {CACHE_PATH}")
            return cache
        print("cache stale (paths changed) -> rebuilding")
    print("embedding images with frozen CLIP (one-time) ...")
    model, preprocess = load_clip(cfg)
    feats = embed_paths(model, preprocess, paths, cfg.device, batch_size=32)
    Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    cache = {"paths": paths, "feats": feats}
    torch.save(cache, CACHE_PATH)
    return cache


def _report(scores: np.ndarray, labels: np.ndarray, t: float) -> None:
    pos, neg = scores[labels == 1], scores[labels == -1]
    acc_pos = float(np.mean(pos <= t)) if len(pos) else float("nan")
    acc_neg = float(np.mean(neg > t)) if len(neg) else float("nan")
    print(f"threshold={t:.4f}  pos accepted={acc_pos:.3f}  neg rejected={acc_neg:.3f}")


if __name__ == "__main__":
    main()
