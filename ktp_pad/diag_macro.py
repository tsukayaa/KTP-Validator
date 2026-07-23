"""
CODE 2c — diagnostic for the macro features. RUN THIS BEFORE RETRAINING.

The dark-region features exist to answer ONE question, and it is not "does the
image have a black area" -- both classes do (spoof = uncropped monitor, genuine =
the photographer's own shadow). The question is:

    among the images that HAVE a dark region, do the features separate
    'display surface' from 'cast shadow'?

So this script reports, in order:

  1. CONTINGENCY  dk_present x label.
     Read this first. If almost no genuine image trips dk_present, the model will
     simply learn "dark region => spoof" and every shadowed genuine photo becomes a
     false reject. That is a DATA problem, not a feature problem, and no amount of
     feature engineering fixes it -- you would need to collect shadowed genuine
     samples. The script says so explicitly when it sees that.

  2. PER-FEATURE AUC on the dk_present==1 subpopulation only.
     This is the honest number for these features. AUC ~0.5 = the feature does not
     discriminate shadow from screen. Also reported over the full set for contrast --
     the full-set number is inflated by dk_present itself and should NOT be quoted.

  3. MASK OVERLAYS (output/macro_masks/*.png).
     Eyeball these. If the detected region is the KTP's own photo box or a block of
     printed text instead of the monitor/shadow, every number above is meaningless.
     Check ~10 per class before trusting anything.

Runs on the macro features ONLY (512 px, no FFT/SRM over full res), so it is fast
enough to sweep the whole dataset.

Run:
    python ktp_pad/diag_macro.py            # whole dataset
    python ktp_pad/diag_macro.py --n-overlay 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_macro import (MACRO_NAMES, DK_NAMES, dark_mask, feat_macro,
                            largest_border_blob, load_macro)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data2"
OUTPUT = ROOT / "output"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = {"positives": "genuine", "negatives": "spoof"}


def collect(cls: str) -> list[Path]:
    """Every image of one class, whether data2 is split into train/test or still flat."""
    seen, out = set(), []
    for d in (DATA / cls, DATA / "train" / cls, DATA / "test" / cls):
        if not d.exists():
            continue
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in EXTS and p.name not in seen:
                seen.add(p.name)
                out.append(p)
    return out


def auc(labels: np.ndarray, x: np.ndarray) -> float:
    """Rank AUC, ties averaged. labels: 1 = spoof. Returns 0.5 when a class is empty."""
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0                                        # average ranks inside each tie group
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def save_overlay(path: Path, dst: Path):
    """Original next to the detected dark region, so the detection can be eyeballed."""
    rgb = load_macro(path)
    V = rgb.mean(axis=2) / 255.0
    c = largest_border_blob(dark_mask(V))
    over = rgb.astype(np.float32) / 255.0
    if c is not None:
        over[c] = 0.6 * over[c] + 0.4 * np.array([1.0, 0.15, 0.15], dtype=np.float32)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].imshow(rgb.astype(np.uint8)); ax[0].set_title("original", fontsize=9)
    ax[1].imshow(np.clip(over, 0, 1))
    ax[1].set_title("detected dark region" if c is not None else "NO dark region", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(path.name, fontsize=8)
    fig.tight_layout()
    fig.savefig(dst, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-overlay", type=int, default=10, help="mask overlays saved per class")
    ap.add_argument("--limit", type=int, default=0, help="cap images per class (0 = all)")
    args = ap.parse_args()

    rows, labels, paths = [], [], []
    for cls, human in CLASSES.items():
        imgs = collect(cls)
        if args.limit:
            imgs = imgs[:args.limit]
        if not imgs:
            sys.exit(f"no images for '{cls}' under {DATA} — put the data in place first")
        print(f"{cls} ({human}): {len(imgs)} images")
        t0 = time.perf_counter()
        for i, p in enumerate(imgs, 1):
            try:
                f, _ = feat_macro(load_macro(p))
            except Exception as e:
                print(f"  skip {p.name}: {e}")
                continue
            rows.append(f)
            labels.append(0 if cls == "positives" else 1)   # 1 = spoof, for AUC direction
            paths.append(p)
            if i % 25 == 0 or i == len(imgs):
                el = time.perf_counter() - t0
                print(f"    {i}/{len(imgs)}  ({el/i*1000:.0f} ms/img, "
                      f"ETA {el/i*(len(imgs)-i):.0f}s)", flush=True)

    X = np.nan_to_num(np.array(rows, dtype=np.float64))
    y = np.array(labels)
    present = X[:, MACRO_NAMES.index("dk_present")] == 1.0

    n_gen, n_spf = int((y == 0).sum()), int((y == 1).sum())
    gp, sp = int((present & (y == 0)).sum()), int((present & (y == 1)).sum())

    L = []
    L.append("# Macro-feature diagnostic\n")
    L.append(f"images: {n_gen} genuine / {n_spf} spoof\n")
    L.append("## 1. Contingency — dk_present x label\n")
    L.append("| | dark region found | none | rate |")
    L.append("|---|---|---|---|")
    L.append(f"| genuine (shadow?) | {gp} | {n_gen-gp} | {gp/max(n_gen,1):.1%} |")
    L.append(f"| spoof (monitor?)  | {sp} | {n_spf-sp} | {sp/max(n_spf,1):.1%} |")
    L.append("")

    usable = gp >= 20 and sp >= 20
    if not usable:
        L.append(f"> **WARNING — subpopulation too small ({gp} genuine / {sp} spoof with a "
                 f"dark region).** The per-feature AUCs below are noise at this N. If the "
                 f"genuine side is the small one, the tree will collapse to "
                 f"`dark region => spoof` and false-reject every shadowed real customer. "
                 f"Fix by collecting shadowed genuine samples, not by tuning features.\n")

    # AUC within the dark-region subpopulation = the number that actually matters
    L.append("## 2. Per-feature AUC\n")
    L.append("`sub` = dk_present==1 only — **this is the honest number**. `all` is inflated "
             "by dk_present itself; do not quote it. 0.5 = no discrimination; distance from "
             "0.5 is what counts (below 0.5 just means the feature points the other way).\n")
    L.append("| feature | AUC sub | AUC all |")
    L.append("|---|---|---|")
    scored = []
    for i, nm in enumerate(MACRO_NAMES):
        a_sub = auc(y[present], X[present, i]) if usable else float("nan")
        a_all = auc(y, X[:, i])
        scored.append((nm, a_sub, a_all))
    key = (lambda t: -abs((t[1] if t[1] == t[1] else 0.5) - 0.5)) if usable \
        else (lambda t: -abs(t[2] - 0.5))
    for nm, a_sub, a_all in sorted(scored, key=key):
        s = "n/a" if a_sub != a_sub else f"{a_sub:.3f}"
        L.append(f"| {nm} | {s} | {a_all:.3f} |")
    L.append("")

    L.append("## 3. Mask overlays\n")
    L.append("`output/macro_masks/` — check that the red region is the monitor / the shadow, "
             "NOT the KTP's own photo box or a text block. If it is the wrong region, every "
             "number above is meaningless.\n")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "macro_diag.md").write_text("\n".join(L), encoding="utf-8")

    mdir = OUTPUT / "macro_masks"
    mdir.mkdir(parents=True, exist_ok=True)
    for cls_label, tag in ((0, "genuine"), (1, "spoof")):
        idx = [i for i in np.nonzero((y == cls_label) & present)[0][:args.n_overlay]]
        for i in idx:
            save_overlay(paths[i], mdir / f"{tag}_{paths[i].stem}.png")
    print(f"\nsaved: {OUTPUT/'macro_diag.md'} , {mdir}/ ({args.n_overlay} per class)")

    print("\n=========== CONTINGENCY (read this first) ===========")
    print(f"genuine with dark region : {gp}/{n_gen} ({gp/max(n_gen,1):.1%})")
    print(f"spoof   with dark region : {sp}/{n_spf} ({sp/max(n_spf,1):.1%})")
    if not usable:
        print("WARNING: subpopulation < 20 on one side — AUCs below are noise.")
    print("\ntop-12 features by |AUC-0.5| within the dark-region subpopulation:")
    for nm, a_sub, a_all in sorted(scored, key=key)[:12]:
        s = "n/a" if a_sub != a_sub else f"{a_sub:.3f}"
        print(f"   {nm:22s} sub {s}   all {a_all:.3f}")


if __name__ == "__main__":
    main()
