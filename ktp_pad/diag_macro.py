"""
CODE 2c — diagnostic for the macro features. RUN THIS BEFORE RETRAINING.

Round 1 of this diagnostic (1171 genuine / 1372 spoof) produced healthy-looking AUCs
for the dk_* group -- and the mask overlays showed they were measuring the wrong
thing. `largest_border_blob` locked onto the room (wall / floor / hoodie / bed on
genuine; monitor, hand, shadow or desk on spoof), not the monitor or the shadow. So
those AUCs were room-background statistics, and the strongest of them ("spoof
surroundings are darker") is a plausible acquisition-environment shortcut rather
than a spoof cue.

That is the whole reason this script draws pictures as well as numbers: **a good AUC
from a mislocalised region is worse than a bad AUC**, because it looks like progress.

Round 2 adds the card-anchored group (cd_*/sr_*): locate the KTP by text density,
then measure the strips immediately around it -- where an uncropped screen actually
sits. dk_* is retained so this run is a direct A/B.

Reports, in order:

  1. CONTINGENCY for both gates.
     `dk_present` (a dark blob exists) and `cd_found` (the card was located).
     `cd_found` is supposed to be class-NEUTRAL: text density is present on a real
     card and on a displayed one alike. If its rate differs much between classes,
     the localiser is itself class-dependent and every sr_* number inherits the bias.

  2. PER-FEATURE AUC, each within its own subpopulation.
     `dk-sub` for the blob group, `cd-sub` for the card group. The `all` column is
     inflated by the gate itself -- do NOT quote it. Distance from 0.5 is what counts.

  3. OVERLAYS (output/macro_masks/*.png), three panels: original | dk_* blob |
     cd_* card (green) + surround strips (blue, darkest in red — the strip every
     sr_dk_* value comes from). `*_nocard.png` are localisation failures.
     Eyeball these BEFORE reading the table, not after.

Runs on the macro features ONLY (512 px, no FFT/SRM over full res), so it is fast
enough to sweep the whole dataset (~100 ms/image).

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
from features_macro import (MACRO_NAMES, dark_mask, feat_macro, find_card, frame_runs,
                            largest_border_blob, load_macro, _strips)

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


def _tint(img, sel, colour, a=0.45):
    img[sel] = (1 - a) * img[sel] + a * np.asarray(colour, dtype=np.float32)


def save_overlay(path: Path, dst: Path):
    """Original | dk_* blob | cd_*/sr_* card + surround | fb_* frame-edge runs.

    Four panels because the three localisations answer different questions and can
    disagree on the same image — the pillarbox case (thin black bars at the frame
    edge, card centred in a bright scene) is invisible to sr_* by construction.
    """
    rgb = load_macro(path)
    V = rgb.mean(axis=2) / 255.0
    base = rgb.astype(np.float32) / 255.0

    blob = base.copy()
    c = largest_border_blob(dark_mask(V))
    if c is not None:
        _tint(blob, c, (1.0, 0.15, 0.15))

    card = base.copy()
    got = find_card(rgb)
    if got is not None:
        cmask, box = got
        _tint(card, cmask, (0.2, 1.0, 0.3), 0.30)                # card = green
        st = _strips(V.shape, box, max(4, int(0.06 * max(V.shape))))
        if st:
            means = {k: float(V[s].mean()) for k, s in st.items()}
            dk = min(means, key=means.get)
            for k, s in st.items():
                sel = np.zeros(V.shape, bool); sel[s] = True
                # darkest strip (the sr_dk_* one) in red, the others in blue
                _tint(card, sel, (1.0, 0.2, 0.2) if k == dk else (0.2, 0.4, 1.0), 0.45)

    frame = base.copy()
    runs, _ = frame_runs(V)
    h, w = V.shape
    spans = {"l": (slice(None), slice(0, runs["l"])), "r": (slice(None), slice(w - runs["r"], w)),
             "t": (slice(0, runs["t"]), slice(None)), "b": (slice(h - runs["b"], h), slice(None))}
    for k, s in spans.items():
        if runs[k] >= 2:
            sel = np.zeros(V.shape, bool); sel[s] = True
            _tint(frame, sel, (1.0, 0.6, 0.0), 0.5)              # runs = orange
    nrun = sum(v >= 2 for v in runs.values())

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    ax[0].imshow(rgb.astype(np.uint8)); ax[0].set_title("original", fontsize=9)
    ax[1].imshow(np.clip(blob, 0, 1))
    ax[1].set_title("dk_*: dark blob" if c is not None else "dk_*: NONE", fontsize=9)
    ax[2].imshow(np.clip(card, 0, 1))
    ax[2].set_title("cd_*/sr_*: card + surround (red = darkest)"
                    if got is not None else "cd_*: CARD NOT FOUND", fontsize=9)
    ax[3].imshow(np.clip(frame, 0, 1))
    ax[3].set_title(f"fb_*: frame-edge runs ({nrun} side(s))" if nrun else
                    "fb_*: no dark edge run", fontsize=9)
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
    found = X[:, MACRO_NAMES.index("cd_found")] == 1.0
    run = X[:, MACRO_NAMES.index("fb_run_max")] > 0.01

    n_gen, n_spf = int((y == 0).sum()), int((y == 1).sum())
    gp, sp = int((present & (y == 0)).sum()), int((present & (y == 1)).sum())
    gf, sf = int((found & (y == 0)).sum()), int((found & (y == 1)).sum())
    gr, sr_ = int((run & (y == 0)).sum()), int((run & (y == 1)).sum())

    L = []
    L.append("# Macro-feature diagnostic\n")
    L.append(f"images: {n_gen} genuine / {n_spf} spoof\n")
    L.append("## 1. Contingency\n")
    L.append("| gate | genuine | spoof | gen rate | spoof rate |")
    L.append("|---|---|---|---|---|")
    L.append(f"| dk_present (dark blob) | {gp} | {sp} | {gp/max(n_gen,1):.1%} | "
             f"{sp/max(n_spf,1):.1%} |")
    L.append(f"| cd_found (card located) | {gf} | {sf} | {gf/max(n_gen,1):.1%} | "
             f"{sf/max(n_spf,1):.1%} |")
    L.append(f"| fb_run_max>0 (dark frame edge) | {gr} | {sr_} | {gr/max(n_gen,1):.1%} | "
             f"{sr_/max(n_spf,1):.1%} |")
    L.append("")
    L.append("`cd_found` is meant to be a NEUTRAL step — it locates the card by print "
             "structure, which a real card and a displayed one both have. If its two rates "
             "differ by more than a few points, the localiser itself is class-dependent and "
             "every sr_* number inherits that bias. Check overlay panel 3 before believing "
             "them.\n")
    L.append("`fb_run_max>0` is NOT meant to be neutral — it is the pillarbox gate. But a "
             "photographer's shadow reaching the frame edge trips it too, so the rate gap "
             "alone proves nothing; the fb_* geometry features are what separate the two.\n")

    usable = gp >= 20 and sp >= 20
    usable_c = gf >= 20 and sf >= 20
    usable_r = gr >= 20 and sr_ >= 20
    if not (usable and usable_c and usable_r):
        L.append(f"> **WARNING — a subpopulation is too small** (dk {gp}/{sp}, cd {gf}/{sf}, "
                 f"fb {gr}/{sr_}). AUCs in that column are noise at this N.\n")

    L.append("## 2. Per-feature AUC\n")
    L.append("Each group read in ITS OWN subpopulation: `dk-sub` = dk_present==1, `cd-sub` = "
             "cd_found==1, `fb-sub` = fb_run_max>0. `all` is inflated by the gate itself — do "
             "not quote it. 0.5 = no discrimination; distance from 0.5 is what counts (below "
             "0.5 just means the feature points the other way).\n")
    L.append("| feature | AUC dk-sub | AUC cd-sub | AUC fb-sub | AUC all |")
    L.append("|---|---|---|---|---|")
    scored = []
    for i, nm in enumerate(MACRO_NAMES):
        a_dk = auc(y[present], X[present, i]) if usable else float("nan")
        a_cd = auc(y[found], X[found, i]) if usable_c else float("nan")
        a_fb = auc(y[run], X[run, i]) if usable_r else float("nan")
        scored.append((nm, a_dk, a_cd, a_fb, auc(y, X[:, i])))

    def dev(v):
        return 0.0 if v != v else abs(v - 0.5)

    def cell(v):
        return "n/a" if v != v else f"{v:.3f}"

    scored.sort(key=lambda t: -max(dev(v) for v in t[1:]))
    for nm, a_dk, a_cd, a_fb, a_all in scored:
        L.append(f"| {nm} | {cell(a_dk)} | {cell(a_cd)} | {cell(a_fb)} | {a_all:.3f} |")
    L.append("")

    L.append("## 3. Mask overlays\n")
    L.append("`output/macro_masks/` — four panels per image:\n")
    L.append("- **panel 2 (`dk_*`)**: the dark blob. On the previous run this locked onto the "
             "room (wall / floor / hoodie / bed) rather than the monitor or the shadow.")
    L.append("- **panel 3 (`cd_*`/`sr_*`)**: green = located card, blue = surround strips, "
             "**red = the darkest strip**, which is the one every `sr_dk_*` value is computed "
             "on. Check the green box is actually the KTP and the red strip is actually the "
             "screen area (on a spoof) — if not, the `cd-sub` column is meaningless too.")
    L.append("- **panel 4 (`fb_*`)**: orange = dark runs scanned inward from the frame edge. "
             "This is the pillarbox case — thin black bars at the far edges with the card "
             "centred in a bright scene, which panel 3 cannot see by construction.")
    L.append("- files ending `_nocard` are cases where localisation failed outright.\n")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "macro_diag.md").write_text("\n".join(L), encoding="utf-8")

    mdir = OUTPUT / "macro_masks"
    mdir.mkdir(parents=True, exist_ok=True)
    n_fail = max(2, args.n_overlay // 4)
    for cls_label, tag in ((0, "genuine"), (1, "spoof")):
        for i in np.nonzero((y == cls_label) & found)[0][:args.n_overlay]:
            save_overlay(paths[i], mdir / f"{tag}_{paths[i].stem}.png")
        for i in np.nonzero((y == cls_label) & ~found)[0][:n_fail]:   # localisation misses
            save_overlay(paths[i], mdir / f"{tag}_{paths[i].stem}_nocard.png")
    print(f"\nsaved: {OUTPUT/'macro_diag.md'} , {mdir}/")

    print("\n=========== CONTINGENCY (read this first) ===========")
    print(f"dk_present  genuine {gp}/{n_gen} ({gp/max(n_gen,1):.1%})   "
          f"spoof {sp}/{n_spf} ({sp/max(n_spf,1):.1%})")
    print(f"cd_found    genuine {gf}/{n_gen} ({gf/max(n_gen,1):.1%})   "
          f"spoof {sf}/{n_spf} ({sf/max(n_spf,1):.1%})   <- should be class-NEUTRAL")
    print(f"fb_run>0    genuine {gr}/{n_gen} ({gr/max(n_gen,1):.1%})   "
          f"spoof {sr_}/{n_spf} ({sr_/max(n_spf,1):.1%})   <- pillarbox gate, not neutral")
    if not (usable and usable_c and usable_r):
        print("WARNING: a subpopulation is < 20 — that column is noise.")
    print("\ntop-15 features by |AUC-0.5| (best of the three subpopulations):")
    for nm, a_dk, a_cd, a_fb, a_all in scored[:15]:
        print(f"   {nm:22s} dk {cell(a_dk)}  cd {cell(a_cd)}  fb {cell(a_fb)}  "
              f"all {a_all:.3f}")


if __name__ == "__main__":
    main()
