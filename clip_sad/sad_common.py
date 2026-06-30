"""
Shared building blocks for the two-version CLIP Deep-SAD KTP validator.

This is the Stage-2 model: it only ever sees images that Stage-1 (zero-shot CLIP
"is a KTP present?") already accepted. Its single job is the hard discrimination:

    KTP-only (front, no person)            -> "yes"   (NORMAL  / label +1)
    selfie+KTP, person+KTP, KTP back,      -> "no"    (ANOMALY / label -1)
    other cards (SIM/ATM/...), unknown ...

Why Deep SAD instead of a binary classifier:
    - Positives are a tight, well-sampled class (~5000). Negatives are few (~500)
      and OPEN-ENDED (many real reject-cases are not in the training set).
    - A binary classifier overfits to the 500 seen negatives and behaves
      undefined on unseen ones.
    - Deep SAD models the POSITIVE manifold (pull positives to a center `c`) and
      pushes the few labeled negatives away. Anything far from `c` at inference
      is rejected -> generalizes to negative types never seen in training.

Both training versions (train_a = frozen/fast, train_b = encoder fine-tune)
share this module so the inference path is identical: embed -> distance to c ->
threshold.

OpenAI CLIP (github.com/openai/CLIP), ViT-B/16, CPU, fully offline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip  # pip install git+https://github.com/openai/CLIP.git


# paths default to folders next to this package so `python -m clip_sad.*` works
# from any working directory. The office-side Copilot can override to absolutes.
_PKG = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  — the office-side Copilot edits these paths/values, nothing else.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SADConfig:
    # --- data layout -----------------------------------------------------------
    #   data_root/<split>/<class>/*.jpg     split in {train, test}
    #   class in {positive, negative, negative_selfie}
    data_root: str = str(_PKG / "data")
    train_split: str = "train"
    test_split: str = "test"
    pos_name: str = "positive"                # KTP-only          (label +1)
    neg_name: str = "negative"                # mixed negatives   (label -1)
    neg_selfie_name: str = "negative_selfie"  # selfie+KTP        (label -1, heavier weight)

    valid_ext: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # --- CLIP checkpoint (OFFLINE: a local path to the ViT-B-16 .pt file) -------
    clip_ckpt: str = str(_PKG / "weights" / "ViT-B-16.pt")
    device: str = "cpu"

    # --- Deep SAD head / objective ---------------------------------------------
    proj_dim: int = 128          # dimension of the SAD embedding z (head output)
    eta: float = 1.0             # weight of the anomaly (negative) term in the loss
    selfie_weight: float = 3.0   # EXTRA multiplier on selfie+KTP negatives (the hard case)
    eps: float = 1e-6            # numerical floor for the inverse-distance term

    # --- optimization ----------------------------------------------------------
    epochs: int = 50
    batch_size: int = 64         # version A can use big batches (cached vectors)
    lr: float = 1e-3
    weight_decay: float = 1e-5   # helps prevent the trivial "collapse to c" solution

    # --- version B (encoder fine-tune) only ------------------------------------
    unfreeze_blocks: int = 2     # number of trailing ViT transformer blocks to train
    lr_encoder: float = 1e-5     # smaller LR for the pretrained encoder than the head

    # --- threshold calibration -------------------------------------------------
    # target recall on positives when no negatives are usable for calibration;
    # if negatives are available we instead pick the threshold that maximizes F1.
    target_pos_recall: float = 0.97

    # --- CPU parallelism (GKE gives 6 vCPU) ------------------------------------
    num_threads: int = 6         # torch intra-op threads for matmul; pin to vCPU count
                                 # (torch auto-detect misreads the cgroup limit in a pod)
    num_workers: int = 4         # DataLoader workers for parallel JPEG decode/preprocess

    random_state: int = 42


# ──────────────────────────────────────────────────────────────────────────────
# Dataset listing
# ──────────────────────────────────────────────────────────────────────────────
def list_images(folder: str, exts: Iterable[str]) -> list[str]:
    p = Path(folder)
    if not p.exists():
        return []
    exts = {e.lower() for e in exts}
    return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in exts)


def build_index(cfg: SADConfig, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (paths, labels, weights) for ONE split ('train' or 'test').

    Reads from  data_root/<split>/{positive,negative,negative_selfie}/.
    labels:  +1 for positives (NORMAL), -1 for negatives (ANOMALY)
    weights:  per-sample loss weight (selfie negatives get cfg.selfie_weight)
    """
    base = Path(cfg.data_root) / split
    pos = list_images(str(base / cfg.pos_name), cfg.valid_ext)
    neg = list_images(str(base / cfg.neg_name), cfg.valid_ext)
    neg_selfie = list_images(str(base / cfg.neg_selfie_name), cfg.valid_ext)

    # de-dup: a selfie image listed in neg_selfie should not be double-counted if
    # neg_dir is a superset. selfie entries win (heavier weight).
    selfie_set = set(neg_selfie)
    neg = [p for p in neg if p not in selfie_set]

    paths = pos + neg + neg_selfie
    labels = np.concatenate([
        np.ones(len(pos), dtype=np.int64),
        -np.ones(len(neg), dtype=np.int64),
        -np.ones(len(neg_selfie), dtype=np.int64),
    ])
    weights = np.concatenate([
        np.ones(len(pos), dtype=np.float32),
        np.ones(len(neg), dtype=np.float32),
        np.full(len(neg_selfie), cfg.selfie_weight, dtype=np.float32),
    ])
    if not pos:
        raise RuntimeError(f"No positive images found in {str(base / cfg.pos_name)!r}")
    return paths, labels, weights


# ──────────────────────────────────────────────────────────────────────────────
# CLIP loading / image embedding (offline)
# ──────────────────────────────────────────────────────────────────────────────
def load_clip(cfg: SADConfig):
    """Load OpenAI CLIP ViT-B/16 from a LOCAL checkpoint (no download).

    clip.load() accepts either a registered model name or a path to a .pt
    checkpoint; we always pass a path so nothing hits the network.
    """
    ckpt = cfg.clip_ckpt
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"CLIP checkpoint not found at {ckpt!r}. Point cfg.clip_ckpt at the "
            f"local ViT-B-16.pt file (offline)."
        )
    model, preprocess = clip.load(ckpt, device=cfg.device)
    model.eval()
    return model, preprocess


class _PathImageDataset(torch.utils.data.Dataset):
    """Decode + preprocess one image per item, so a DataLoader can parallelize it."""

    def __init__(self, paths: list[str], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.preprocess(img)


@torch.no_grad()
def embed_paths(model, preprocess, paths: list[str], device: str,
                batch_size: int = 32, num_workers: int = 0,
                log_every: int = 500) -> torch.Tensor:
    """Encode a list of image paths into L2-normalized CLIP features [N, 512].

    Image decode + preprocess run in `num_workers` parallel workers (the serial
    PIL loop was the dominant cost of version A on a multi-core CPU). Order is
    preserved (shuffle=False). Used by version A to cache features once, and by
    predict.py at inference.
    """
    from torch.utils.data import DataLoader
    ds = _PathImageDataset(paths, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers,
                    persistent_workers=num_workers > 0)
    feats = []
    done = 0
    for x in dl:
        x = x.to(device)
        f = F.normalize(model.encode_image(x).float(), dim=-1)
        feats.append(f.cpu())
        done += len(x)
        if log_every and done % log_every < batch_size:
            print(f"  embedded {done}/{len(paths)}")
    return torch.cat(feats, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Model pieces
# ──────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """Maps a CLIP feature to the SAD embedding z.

    NOTE on Deep SVDD/SAD collapse: bias-free linear layers and weight decay are
    deliberate — they discourage the trivial all-to-center solution. Do not add
    a final saturating activation (it enables collapse).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class FinetuneSAD(nn.Module):
    """Version-B model: trainable CLIP visual encoder (last N blocks) + head.

    forward(images) -> z   (the SAD embedding)
    """

    def __init__(self, clip_model, proj_dim: int, unfreeze_blocks: int):
        super().__init__()
        self.clip = clip_model
        self.head = ProjectionHead(clip_model.visual.output_dim, proj_dim)

        # freeze everything, then unfreeze the last `unfreeze_blocks` ViT blocks
        for p in self.clip.parameters():
            p.requires_grad_(False)
        blocks = self.clip.visual.transformer.resblocks
        for blk in blocks[-unfreeze_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        # also let the post-LN adapt
        for p in self.clip.visual.ln_post.parameters():
            p.requires_grad_(True)

    def encode(self, images):
        f = self.clip.encode_image(images).float()
        return F.normalize(f, dim=-1)

    def forward(self, images):
        return self.head(self.encode(images))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ──────────────────────────────────────────────────────────────────────────────
# Deep SAD objective
# ──────────────────────────────────────────────────────────────────────────────
def init_center(z: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """Center c = mean of initial embeddings; nudge near-zero dims away from 0
    (standard Deep SVDD trick so no coordinate is trivially satisfied)."""
    c = z.mean(dim=0)
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c >= 0)] = eps
    return c


def deep_sad_loss(z, labels, weights, c, eta: float, eps: float):
    """Deep SAD loss.

    NORMAL  (label +1): pull toward center      -> ||z - c||^2
    ANOMALY (label -1): push away from center   -> eta * w * (||z - c||^2)^-1
    """
    dist = ((z - c) ** 2).sum(dim=1)
    normal_term = dist
    anomaly_term = eta * weights * (dist + eps) ** (-1.0)
    per_sample = torch.where(labels == 1, normal_term, anomaly_term)
    return per_sample.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Scoring + threshold calibration
# ──────────────────────────────────────────────────────────────────────────────
def scores_from_z(z: torch.Tensor, c: torch.Tensor) -> np.ndarray:
    """Anomaly score = squared distance to center. Small = looks like a KTP-only."""
    return ((z - c) ** 2).sum(dim=1).cpu().numpy()


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray,
                        target_pos_recall: float) -> float:
    """Pick the decision threshold on the anomaly score.

    label==+1 means positive (should score LOW / be accepted).
    If labeled negatives exist, choose the threshold that maximizes F1 of the
    "accept" decision. Otherwise fall back to the positive-recall quantile.
    """
    pos = scores[labels == 1]
    neg = scores[labels == -1]

    if len(neg) == 0:
        # no negatives to trade off against -> accept the bulk of positives
        return float(np.quantile(pos, target_pos_recall))

    candidates = np.unique(np.concatenate([pos, neg]))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        tp = np.sum(pos <= t)          # positive correctly accepted
        fp = np.sum(neg <= t)          # negative wrongly accepted
        fn = np.sum(pos > t)           # positive wrongly rejected
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ──────────────────────────────────────────────────────────────────────────────
# Save / load a trained bundle  (consumed by predict.py for BOTH versions)
# ──────────────────────────────────────────────────────────────────────────────
def save_bundle(path: str, *, mode: str, cfg: SADConfig, center: torch.Tensor,
                threshold: float, head: nn.Module, visual_state: dict | None = None):
    """mode is 'A' (frozen head) or 'B' (encoder fine-tune)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "mode": mode,
        "proj_dim": cfg.proj_dim,
        "center": center.detach().cpu(),
        "threshold": float(threshold),
        "head_state": head.state_dict(),
        # version B also needs the fine-tuned visual weights to reproduce z
        "visual_state": visual_state,
        "config": asdict(cfg),
    }
    torch.save(bundle, path)
    # human-readable sidecar
    Path(path).with_suffix(".meta.json").write_text(json.dumps(
        {"mode": mode, "threshold": float(threshold), "proj_dim": cfg.proj_dim},
        indent=2,
    ))


def load_bundle(path: str):
    return torch.load(path, map_location="cpu")
