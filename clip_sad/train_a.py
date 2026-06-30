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


def main():
    cfg = SADConfig()
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.set_num_threads(cfg.num_threads)   # use all GKE vCPUs for the CLIP matmul

    # ── 1. index + cache CLIP features for BOTH splits (one encoder load) ─────
    paths_tr, labels_tr, weights_tr = build_index(cfg, cfg.train_split)
    print(f"train: {int((labels_tr==1).sum())} pos / {int((labels_tr==-1).sum())} neg "
          f"({len(paths_tr)} total)")
    cache_tr, clip_bundle = _load_or_build_cache(cfg, paths_tr, cfg.train_split)
    feats_tr = cache_tr["feats"]                  # [N, 512] L2-normalized, frozen

    paths_te, labels_te, _ = build_index(cfg, cfg.test_split)
    print(f"test : {int((labels_te==1).sum())} pos / {int((labels_te==-1).sum())} neg "
          f"({len(paths_te)} total)")
    cache_te, _ = _load_or_build_cache(cfg, paths_te, cfg.test_split, clip_bundle)
    feats_te = cache_te["feats"]

    # ── 2. init head + center (center from TRAIN forward) ─────────────────────
    head = ProjectionHead(feats_tr.shape[1], cfg.proj_dim)
    with torch.no_grad():
        z0 = head(feats_tr)
    center = init_center(z0)

    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ds = TensorDataset(feats_tr, torch.from_numpy(labels_tr), torch.from_numpy(weights_tr))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # ── 3. train head with Deep SAD loss (TRAIN only) ─────────────────────────
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

    # ── 4. calibrate threshold on TRAIN, evaluate on held-out TEST ────────────
    head.eval()
    with torch.no_grad():
        scores_tr = scores_from_z(head(feats_tr), center)
        scores_te = scores_from_z(head(feats_te), center)
    threshold = calibrate_threshold(scores_tr, labels_tr, cfg.target_pos_recall)
    print("[train]", end=" "); _report(scores_tr, labels_tr, threshold)
    print("[test ]", end=" "); _report(scores_te, labels_te, threshold)

    save_bundle(OUT_PATH, mode="A", cfg=cfg, center=center,
                threshold=threshold, head=head, visual_state=None)
    print(f"saved -> {OUT_PATH}")


def _load_or_build_cache(cfg: SADConfig, paths: list[str], split: str,
                         clip_bundle=None) -> tuple[dict, tuple | None]:
    """Cache frozen CLIP features per split. Reuses an already-loaded CLIP
    (clip_bundle) so train+test don't load the encoder twice."""
    cache_path = str(_PKG / "cache" / f"clip_features_A_{split}.pt")
    if Path(cache_path).exists():
        cache = torch.load(cache_path, map_location="cpu")
        if cache.get("paths") == paths:
            print(f"using cached features: {cache_path}")
            return cache, clip_bundle
        print(f"cache stale ({split}, paths changed) -> rebuilding")
    print(f"embedding {split} images with frozen CLIP (one-time) ...")
    if clip_bundle is None:
        clip_bundle = load_clip(cfg)
    model, preprocess = clip_bundle
    feats = embed_paths(model, preprocess, paths, cfg.device,
                        batch_size=32, num_workers=cfg.num_workers)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    cache = {"paths": paths, "feats": feats}
    torch.save(cache, cache_path)
    return cache, clip_bundle


def _report(scores: np.ndarray, labels: np.ndarray, t: float) -> None:
    pos, neg = scores[labels == 1], scores[labels == -1]
    acc_pos = float(np.mean(pos <= t)) if len(pos) else float("nan")
    acc_neg = float(np.mean(neg > t)) if len(neg) else float("nan")
    print(f"threshold={t:.4f}  pos accepted={acc_pos:.3f}  neg rejected={acc_neg:.3f}")


if __name__ == "__main__":
    main()
