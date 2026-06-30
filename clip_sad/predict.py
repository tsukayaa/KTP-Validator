"""
Inference for BOTH versions (A frozen-head, B encoder-fine-tune).

The decision is identical regardless of version:
    z      = head(normalize(CLIP.encode_image(x)))
    score  = ||z - center||^2          (small = looks like a KTP-only)
    label  = "yes" if score <= threshold else "no"

This is the Stage-2 reject gate. Feed it only images Stage-1 already accepted as
"a KTP is present".

Usage:
    python -m clip_sad.predict path/to/image.jpg --model models/clip_sad_B.pt
    python -m clip_sad.predict path/to/folder   --model models/clip_sad_A.pt
    # override the offline CLIP checkpoint if it lives elsewhere on this machine:
    python -m clip_sad.predict img.jpg --model models/clip_sad_B.pt --clip_ckpt /opt/ViT-B-16.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from clip_sad.sad_common import SADConfig, load_clip, ProjectionHead, load_bundle


class SADPredictor:
    def __init__(self, model_path: str, clip_ckpt: str | None = None):
        bundle = load_bundle(model_path)
        cfg = SADConfig(**bundle["config"])
        if clip_ckpt:
            cfg.clip_ckpt = clip_ckpt

        self.mode = bundle["mode"]
        self.center = bundle["center"]
        self.threshold = bundle["threshold"]
        self.cfg = cfg

        self.clip, self.preprocess = load_clip(cfg)
        if self.mode == "B":
            # restore the fine-tuned visual weights on top of the base CLIP
            self.clip.visual.load_state_dict(bundle["visual_state"])
            self.clip.eval()

        in_dim = self.clip.visual.output_dim
        self.head = ProjectionHead(in_dim, bundle["proj_dim"])
        self.head.load_state_dict(bundle["head_state"])
        self.head.eval()

    @torch.no_grad()
    def score(self, image_path: str) -> float:
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.cfg.device)
        feat = F.normalize(self.clip.encode_image(x).float(), dim=-1)
        z = self.head(feat)
        return float(((z - self.center) ** 2).sum().item())

    def predict(self, image_path: str) -> dict:
        s = self.score(image_path)
        return {
            "class": "yes" if s <= self.threshold else "no",
            "score": s,
            "threshold": self.threshold,
            "mode": self.mode,
        }


def _print(image_path: str, r: dict, ms: float) -> None:
    bar = "=" * 44
    print(bar)
    print(f"  Image     : {Path(image_path).name}")
    print(f"  Prediction: {r['class']}   (KTP-only={r['class']=='yes'})")
    print(f"  Score     : {r['score']:.4f}   threshold={r['threshold']:.4f}")
    print(f"  Version   : {r['mode']}")
    print(f"  Time      : {ms:.0f}ms")
    print(bar)


def main():
    ap = argparse.ArgumentParser(description="CLIP Deep-SAD KTP-only reject gate (Stage 2)")
    ap.add_argument("path", help="image file or a folder of images")
    ap.add_argument("--model", default="models/clip_sad_B.pt", help="trained bundle (.pt)")
    ap.add_argument("--clip_ckpt", default=None, help="override offline CLIP .pt path")
    args = ap.parse_args()

    predictor = SADPredictor(args.model, args.clip_ckpt)

    target = Path(args.path)
    if target.is_dir():
        exts = predictor.cfg.valid_ext
        images = sorted(p for p in target.rglob("*") if p.suffix.lower() in exts)
        n_yes = 0
        for img in images:
            t0 = time.perf_counter()
            r = predictor.predict(str(img))
            ms = (time.perf_counter() - t0) * 1000
            n_yes += r["class"] == "yes"
            print(f"{r['class']:>3}  score={r['score']:8.4f}  {img.name}  ({ms:.0f}ms)")
        print(f"\n{n_yes}/{len(images)} predicted KTP-only (yes)")
    else:
        t0 = time.perf_counter()
        r = predictor.predict(str(target))
        _print(str(target), r, (time.perf_counter() - t0) * 1000)


if __name__ == "__main__":
    main()
