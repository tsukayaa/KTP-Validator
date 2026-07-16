"""
Feature visualization: genuine KTP vs 2 spoof subtypes (bezel-visible / pendar-glare).

Produces one panel PNG — 3 rows (genuine / bezel / pendar) x 4 cols (original |
FFT spectrum | SRM residual | specular/glare mask) — for the "features used"
slide, non-technical audience. Reuses the EXACT feature functions from
train_pad.py (load_gray, load_rgb, SRM_KERNELS) so the picture matches what the
model actually sees, not a redrawn approximation.

NOTE: data2/negatives is NOT split by spoof subtype (bezel vs pendar are just
two visual symptoms of the same "photo of a screen" act — no folder tells you
which is which). Auto-pick below grabs 2 arbitrary negatives; for an accurate
presentation, LOOK at your negatives and pass the 2 file paths explicitly:

    python ktp_pad/viz_features.py \
        --genuine data2/test/positives/xxx.jpg \
        --bezel   data2/test/negatives/yyy.jpg \
        --pendar  data2/test/negatives/zzz.jpg

Writes: output/feature_panel.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pad as tp

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def _pick(cls: str, skip: set[Path] = frozenset()) -> Path:
    for d in (ROOT / "data2" / "test" / cls, ROOT / "data2" / cls, ROOT / "data2" / "train" / cls):
        if d.exists():
            imgs = [p for p in tp.list_images(d) if p not in skip]
            if imgs:
                return imgs[0]
    raise FileNotFoundError(
        f"no image found for class '{cls}' under {ROOT/'data2'} (checked test/ , flat, train/)"
    )


def fft_spectrum(gray: np.ndarray) -> np.ndarray:
    g = gray - gray.mean()
    h, w = g.shape
    g = g * (np.hanning(h)[:, None] * np.hanning(w)[None, :])
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))


def srm_grid_residual(gray: np.ndarray) -> np.ndarray:
    kernel = tp.SRM_KERNELS[4]          # KV square kernel -> screen pixel-grid cue
    res = ndimage.convolve(gray / 255.0, kernel, mode="reflect")
    return np.clip(res, -0.3, 0.3)


def specular_overlay(rgb: np.ndarray) -> np.ndarray:
    """Same mask logic as feat_specular in train_pad.py, drawn as a red overlay ('pendar' glare)."""
    mx = rgb.max(axis=2); mn = rgb.min(axis=2)
    V = mx / 255.0
    S = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)
    mask = (V > 0.85) & (S < 0.15)
    base = np.stack([rgb.mean(axis=2) / 255.0] * 3, axis=2)
    base[mask] = [1.0, 0.15, 0.15]
    return base


def panel_row(ax_row, path: Path, label: str):
    gray = tp.load_gray(path)
    rgb = tp.load_rgb(path).astype(np.uint8)

    ax_row[0].imshow(rgb)
    ax_row[0].set_ylabel(label, fontsize=13, fontweight="bold")
    ax_row[0].set_title("original", fontsize=11)

    ax_row[1].imshow(fft_spectrum(gray), cmap="inferno")
    ax_row[1].set_title("FFT spectrum (moire)", fontsize=11)

    ax_row[2].imshow(srm_grid_residual(gray), cmap="gray")
    ax_row[2].set_title("SRM residual (screen grid)", fontsize=11)

    ax_row[3].imshow(specular_overlay(rgb))
    ax_row[3].set_title("specular mask (glare)", fontsize=11)

    for a in ax_row:
        a.set_xticks([]); a.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genuine", type=Path, default=None)
    ap.add_argument("--bezel", type=Path, default=None)
    ap.add_argument("--pendar", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUTPUT / "feature_panel.png")
    args = ap.parse_args()

    gpath = args.genuine or _pick("positives")
    bpath = args.bezel or _pick("negatives")
    ppath = args.pendar or _pick("negatives", skip={bpath})

    if args.bezel is None or args.pendar is None:
        print("WARNING: --bezel/--pendar not given -> auto-picked ARBITRARY negatives.")
        print("         data2/negatives is not subtype-labeled; verify these are actually")
        print("         a bezel-visible shot and a glare/pendar shot, or pass paths explicitly.")

    print(f"genuine: {gpath}")
    print(f"bezel  : {bpath}")
    print(f"pendar : {ppath}")

    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    panel_row(axes[0], gpath, "Genuine")
    panel_row(axes[1], bpath, "Spoof (bezel)")
    panel_row(axes[2], ppath, "Spoof (pendar)")
    fig.suptitle("Forensic features: genuine vs screen-replay (bezel / pendar)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
