"""
STAGE 1  —  zero-shot CLIP gate:  "is a KTP present in this image?"

This runs BEFORE the Deep-SAD Stage-2 model. Its only job is to drop images
with no KTP at all, so Stage 2 only ever sees KTP-containing images and can
specialize on the hard boundary (KTP-only vs selfie+KTP / back / other cards).

Prompt set provenance
---------------------
These are the v2.5 prompts from baru.md Appendix B — the last stable set from
*before* the requirement flipped to rejecting people. At this stage selfie+KTP
is intentionally a "yes" (a KTP IS present) -> it passes Stage 1 and is rejected
later by Stage 2. That is the whole point of the two-phase split.

Design notes (from baru.md):
  - `yes` stresses the PHYSICAL FORM ("small plastic card", "credit-card sized",
    "with a photo") to separate a KTP from A4 paper documents.
  - `no` names the proven document hard-negatives explicitly (Kartu Keluarga,
    akta kelahiran) AND keeps the generalizing workhorse "a large paper document
    or certificate, not a plastic card" so KK/akte/ijazah/NPWP get rejected here
    and never reach Stage 2.
  - Aggregation is MAX per pool (highest single similarity), not mean.
  - Encode the text prompts ONCE at startup; only image encoding scales per image.

Note: SIM/ATM/other small plastic cards may still pass Stage 1 (CLIP can't cleanly
tell them from a KTP at 224px) — that is expected; Stage 2 rejects them.

OpenAI CLIP ViT-B/16, CPU, offline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import clip

from clip_sad.sad_common import SADConfig, load_clip


# ──────────────────────────────────────────────────────────────────────────────
# Phase-1 prompt pools  (baru.md Appendix B — pre-selfie-attack "KTP present?")
# ──────────────────────────────────────────────────────────────────────────────
YES_PROMPTS = [
    "a photo of an Indonesian ID card (KTP), a small plastic card",
    "a close-up of a KTP card, credit-card sized with a photo",
    "an Indonesian KTP identity card lying on a surface",
    "a hand holding a small Indonesian ID card (KTP)",
    "a person taking a selfie while showing a KTP card",
    "a small plastic ID card with a photo, clearly visible",
]

NO_PROMPTS = [
    "a random photo that does not contain any ID card",
    "a photo of an everyday scene with no identity card",
    "a selfie of a person without any identity card",
    "an Indonesian family card document (Kartu Keluarga) on paper",
    "an Indonesian birth certificate (akta kelahiran) on paper",
    "a large paper document or certificate, not a plastic card",
    "a random image unrelated to ID cards",
]


class Stage1Gate:
    """Zero-shot "is a KTP present?" gate. Text prompts encoded once at init."""

    def __init__(self, cfg: SADConfig | None = None):
        self.cfg = cfg or SADConfig()
        self.model, self.preprocess = load_clip(self.cfg)
        self.yes_text = self._encode_text(YES_PROMPTS)
        self.no_text = self._encode_text(NO_PROMPTS)

    @torch.no_grad()
    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        tokens = clip.tokenize(prompts).to(self.cfg.device)
        t = self.model.encode_text(tokens).float()
        return F.normalize(t, dim=-1)              # [P, 512]

    @torch.no_grad()
    def is_ktp_present(self, image_path: str) -> dict:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.cfg.device)
        feat = F.normalize(self.model.encode_image(x).float(), dim=-1)  # [1, 512]

        yes_score = (feat @ self.yes_text.T).max().item()   # MAX per pool
        no_score = (feat @ self.no_text.T).max().item()
        present = yes_score > no_score
        return {"present": present, "yes_score": yes_score, "no_score": no_score}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage-1 zero-shot: is a KTP present?")
    ap.add_argument("image_path")
    ap.add_argument("--clip_ckpt", default=None)
    args = ap.parse_args()

    cfg = SADConfig()
    if args.clip_ckpt:
        cfg.clip_ckpt = args.clip_ckpt
    gate = Stage1Gate(cfg)
    r = gate.is_ktp_present(args.image_path)
    verdict = "KTP present -> pass to Stage 2" if r["present"] else "no KTP -> reject"
    print(f"{verdict}   (yes={r['yes_score']:.4f}  no={r['no_score']:.4f})")
