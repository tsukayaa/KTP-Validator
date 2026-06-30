"""
VERSION B  —  FULL fine-tune  (~9 h on 6 CPU, ~1-2 h on ~30 vCPU)

The CLIP ViT-B/16 visual encoder (last N transformer blocks) is fine-tuned
end-to-end with the Deep-SAD objective, so the representation itself adapts to
separate selfie+KTP from KTP-only — the case version A cannot reach. The
selfie+KTP negatives carry a heavier loss weight (cfg.selfie_weight) so the
gradient is steered at the case you care about most.

Because the encoder changes every step, images are re-forwarded each epoch
(no embedding cache). This is the slow path — prefer more vCPUs on GKE.

Run:
    python -m clip_sad.train_b
Output:
    models/clip_sad_B.pt   (load with predict.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from clip_sad.sad_common import (
    SADConfig, build_index, load_clip,
    FinetuneSAD, init_center, deep_sad_loss,
    scores_from_z, calibrate_threshold, save_bundle,
)

OUT_PATH = str(Path(__file__).resolve().parent / "models" / "clip_sad_B.pt")


class ImageDataset(Dataset):
    def __init__(self, paths, labels, weights, preprocess):
        self.paths = paths
        self.labels = labels
        self.weights = weights
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.preprocess(img)
        return x, self.labels[i], self.weights[i]


def main():
    cfg = SADConfig()
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    torch.set_num_threads(max(1, torch.get_num_threads()))  # use the CPU cores GKE gives us

    paths, labels, weights = build_index(cfg)
    print(f"dataset: {int((labels==1).sum())} pos / {int((labels==-1).sum())} neg "
          f"({len(paths)} total)")

    model, preprocess = load_clip(cfg)
    net = FinetuneSAD(model, cfg.proj_dim, cfg.unfreeze_blocks).to(cfg.device)

    ds = ImageDataset(paths, torch.from_numpy(labels), torch.from_numpy(weights), preprocess)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    # ── center: one forward pass over POSITIVES only, before training ─────────
    center = _init_center_from_positives(net, ds, labels, cfg)

    # two param groups: tiny LR for the pretrained encoder, normal LR for the head
    enc_params = [p for n, p in net.named_parameters()
                  if p.requires_grad and not n.startswith("head.")]
    head_params = [p for p in net.head.parameters() if p.requires_grad]
    opt = torch.optim.Adam([
        {"params": enc_params, "lr": cfg.lr_encoder},
        {"params": head_params, "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)

    # ── train ────────────────────────────────────────────────────────────────
    for epoch in range(cfg.epochs):
        net.train()
        running, n = 0.0, 0
        for xb, lb, wb in dl:
            xb = xb.to(cfg.device)
            lb = lb.to(cfg.device)
            wb = wb.to(cfg.device)
            opt.zero_grad()
            z = net(xb)
            loss = deep_sad_loss(z, lb, wb, center, cfg.eta, cfg.eps)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            n += len(xb)
        print(f"epoch {epoch:3d}  loss {running/n:.4f}")

    # ── calibrate threshold (against the SAME fixed center used in training) ──
    scores, labels_eval = _score_all(net, ds, center, cfg)
    threshold = calibrate_threshold(scores, labels_eval, cfg.target_pos_recall)
    _report(scores, labels_eval, threshold)

    save_bundle(OUT_PATH, mode="B", cfg=cfg, center=center, threshold=threshold,
                head=net.head, visual_state=net.clip.visual.state_dict())
    print(f"saved -> {OUT_PATH}")


@torch.no_grad()
def _init_center_from_positives(net, ds, labels, cfg) -> torch.Tensor:
    net.eval()
    pos_idx = np.where(labels == 1)[0]
    zs = []
    dl = DataLoader(_Subset(ds, pos_idx), batch_size=cfg.batch_size)
    for xb, _, _ in dl:
        zs.append(net(xb.to(cfg.device)).cpu())
    z = torch.cat(zs, dim=0)
    return init_center(z)


@torch.no_grad()
def _score_all(net, ds, center, cfg):
    """Score every image against the FIXED center used during training.
    (Deep SVDD/SAD keeps c constant from init — do not recompute it here.)"""
    net.eval()
    dl = DataLoader(ds, batch_size=cfg.batch_size)
    zs, ls = [], []
    for xb, lb, wb in dl:
        zs.append(net(xb.to(cfg.device)).cpu())
        ls.append(lb)
    z = torch.cat(zs, dim=0)
    labels = torch.cat(ls, dim=0).numpy()
    scores = scores_from_z(z, center)
    return scores, labels


class _Subset(Dataset):
    def __init__(self, ds, idx):
        self.ds, self.idx = ds, list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds[self.idx[i]]


def _report(scores, labels, t):
    pos, neg = scores[labels == 1], scores[labels == -1]
    acc_pos = float(np.mean(pos <= t)) if len(pos) else float("nan")
    acc_neg = float(np.mean(neg > t)) if len(neg) else float("nan")
    print(f"threshold={t:.4f}  pos accepted={acc_pos:.3f}  neg rejected={acc_neg:.3f}")


if __name__ == "__main__":
    main()
