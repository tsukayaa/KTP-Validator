"""
CODE 2 — KTP screen-replay detector: forensic features -> LightGBM.

WHY NOT CLIP: screen-replay ("pendar"/moiré/bezel) is a LOW-LEVEL TEXTURE signal.
CLIP resizes to 224x224 and is trained to be INVARIANT to resize/JPEG/color — exactly
the recapture cues we need — so a CLIP fine-tune is structurally blind to it (that is why
the earlier CLIP run stalled at F1~0.74). Here we compute the texture forensics CLIP throws
away (FFT moire peaks, SRM residuals, specular/glare, LBP micro-texture, color banding,
JPEG double-compression blockiness) and feed a small gradient-boosted tree. No pretrained
weights, CPU, fully offline.

Reads:   data2/train/{positives,negatives}   data2/test/{positives,negatives}
         positives = KTP asli (label 1)   negatives = spoof (label 0)
Writes:  models/pad_lgbm.txt   models/pad_meta.json
         output/results.json                    (per-file scores + confusion + metrics)
         output/positives/  output/negatives/    (test images sorted by PREDICTION)

Run:
    python ktp_pad/split_dataset.py     # once
    python ktp_pad/train_pad.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.stats import kurtosis
import lightgbm as lgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data2"
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 1024          # center-crop (NOT resize) large images to bound compute
SEED = 42
CLIP_BASELINE_F1 = 0.74  # the number to beat


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING  (native pixels — never resize; resize destroys the texture signal)
# ══════════════════════════════════════════════════════════════════════════════
def _center_crop(a: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = a.shape[:2]
    if max(h, w) <= max_side:
        return a
    top = max(0, (h - max_side) // 2)
    left = max(0, (w - max_side) // 2)
    return a[top:top + min(h, max_side), left:left + min(w, max_side)]


def load_gray(path: Path) -> np.ndarray:
    return _center_crop(np.asarray(Image.open(path).convert("L"), dtype=np.float32))


def load_rgb(path: Path) -> np.ndarray:
    return _center_crop(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUPS
# ══════════════════════════════════════════════════════════════════════════════
def feat_fft(gray, NR=32, NA=12):
    """FFT log-magnitude -> radial + angular profile + moire peak stats. STRONGEST cue."""
    g = gray - gray.mean()
    h, w = g.shape
    g = g * (np.hanning(h)[:, None] * np.hanning(w)[None, :])   # window -> less edge leakage
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = float(r.max()) + 1e-6
    rn = np.clip((r / rmax * NR).astype(int), 0, NR - 1)
    prof = np.array([mag[rn == i].mean() if np.any(rn == i) else 0.0 for i in range(NR)])
    prof_n = prof / (prof.max() + 1e-6)
    ratio = prof[NR // 2:].sum() / (prof[:NR // 2].sum() + 1e-6)         # high/low energy
    detr = prof - ndimage.uniform_filter1d(prof, size=5, mode="nearest")  # detrend
    peaks, props = find_peaks(detr, prominence=detr.std() * 0.5 + 1e-6)
    npk = len(peaks)
    mxprom = float(props["prominences"].max()) if npk else 0.0
    mnprom = float(props["prominences"].mean()) if npk else 0.0
    ring = (r > 0.25 * rmax) & (r < 0.75 * rmax)
    ang = np.arctan2(y - cy, x - cx) + np.pi
    an = (ang / (2 * np.pi) * NA).astype(int) % NA
    aprof = np.array([mag[ring & (an == i)].mean() if np.any(ring & (an == i)) else 0.0
                      for i in range(NA)])
    a_var = float(aprof.var())
    a_ratio = (aprof.max() + 1e-6) / (aprof.mean() + 1e-6)
    feats = list(prof_n) + [ratio, npk, mxprom, mnprom, a_var, a_ratio] + \
        list(aprof / (aprof.max() + 1e-6))
    names = [f"fft_r{i}" for i in range(NR)] + \
        ["fft_hilo", "fft_npeaks", "fft_maxprom", "fft_meanprom", "fft_angvar", "fft_angratio"] + \
        [f"fft_a{i}" for i in range(NA)]
    return feats, names


def _k(a):
    return np.array(a, dtype=np.float32)


SRM_KERNELS = [
    _k([[0, 0, 0], [0, -1, 1], [0, 0, 0]]),                        # 1st deriv x
    _k([[0, 0, 0], [0, -1, 0], [0, 1, 0]]),                        # 1st deriv y
    _k([[0, 0, 0], [1, -2, 1], [0, 0, 0]]),                        # 2nd deriv x
    _k([[0, 1, 0], [0, -2, 0], [0, 1, 0]]),                        # 2nd deriv y
    _k([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4.0,              # KV square (screen grid)
]


def feat_srm(gray):
    """High-pass residuals -> stats + quantized histogram. Targets screen pixel-grid noise."""
    feats, names = [], []
    g = gray / 255.0
    for ki, k in enumerate(SRM_KERNELS):
        res = ndimage.convolve(g, k, mode="reflect")
        rc = np.clip(res, -1, 1)
        kv = float(kurtosis(rc.ravel(), fisher=True, bias=False)) if rc.std() > 1e-6 else 0.0
        feats += [float(rc.std()), float(np.abs(rc).mean()), kv]
        hist, _ = np.histogram(np.clip(res, -0.3, 0.3), bins=5, range=(-0.3, 0.3), density=True)
        feats += list(hist * 0.1)
        names += [f"srm{ki}_std", f"srm{ki}_mad", f"srm{ki}_kurt"] + \
                 [f"srm{ki}_h{j}" for j in range(5)]
    return feats, names


def feat_specular(rgb):
    """Bright + low-saturation blobs = display glare ('pendar')."""
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    V = mx / 255.0
    S = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)
    mask = (V > 0.85) & (S < 0.15)
    lab, nb = ndimage.label(mask)
    if nb > 0:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, nb + 1)) / mask.size
        mxb, mnb = float(sizes.max()), float(sizes.mean())
    else:
        mxb = mnb = 0.0
    gy, gx = np.gradient(V)
    gm = np.sqrt(gx ** 2 + gy ** 2)
    grad = float(gm[mask].mean()) if mask.any() else 0.0
    feats = [float(mask.mean()), nb / 1000.0, mxb, mnb, grad]
    names = ["spec_area", "spec_nblob", "spec_maxblob", "spec_meanblob", "spec_grad"]
    return feats, names


def _lbp_uniform(gray, R=1):
    """Rotation-invariant uniform LBP (P=8), 10 bins. Screen vs paper/PVC micro-texture."""
    P = 8
    offs = [(-R, 0), (-R, R), (0, R), (R, R), (R, 0), (R, -R), (0, -R), (-R, -R)]
    bits = np.stack([(np.roll(np.roll(gray, -dy, 0), -dx, 1) >= gray).astype(np.int8)
                     for dy, dx in offs], axis=0)
    trans = np.zeros(gray.shape, dtype=np.int16)
    for i in range(P):
        trans += (bits[i] != bits[(i + 1) % P]).astype(np.int16)
    ones = bits.sum(axis=0)
    label = np.where(trans <= 2, ones, 9)
    label = label[R:-R, R:-R]                       # drop wrap-around border
    hist, _ = np.histogram(label, bins=np.arange(11), density=True)
    return list(hist)


def feat_lbp(gray):
    feats = _lbp_uniform(gray, 1) + _lbp_uniform(gray, 2)
    names = [f"lbp1_{i}" for i in range(10)] + [f"lbp2_{i}" for i in range(10)]
    return feats, names


def feat_color(rgb):
    """Histogram entropy + empty-bin count (banding) + flat-gradient fraction. Display gamut."""
    feats, names = [], []
    for ci, ch in enumerate("rgb"):
        v = rgb[:, :, ci].astype(np.int32)
        hist, _ = np.histogram(v, bins=256, range=(0, 255))
        p = hist / (hist.sum() + 1e-6)
        ent = -(p[p > 0] * np.log2(p[p > 0])).sum()
        feats += [float(ent / 8.0), float((hist == 0).mean())]
        names += [f"col_{ch}_entropy", f"col_{ch}_emptybins"]
    gy, gx = np.gradient(rgb.mean(axis=2))
    gm = np.sqrt(gx ** 2 + gy ** 2)
    feats += [float((gm < 1.0).mean())]
    names += ["col_flatgrad"]
    return feats, names


def feat_blockiness(gray):
    """8x8 block-boundary discontinuity vs within-block = JPEG double-compression cue."""
    h, w = gray.shape
    dh = np.abs(np.diff(gray, axis=1))
    cols = np.arange(7, w - 1, 8)
    within_c = np.setdiff1d(np.arange(0, w - 1), cols)
    b_h = dh[:, cols].mean() if len(cols) else 0.0
    w_h = dh[:, within_c].mean() if len(within_c) else 1e-6
    dv = np.abs(np.diff(gray, axis=0))
    rows = np.arange(7, h - 1, 8)
    within_r = np.setdiff1d(np.arange(0, h - 1), rows)
    b_v = dv[rows, :].mean() if len(rows) else 0.0
    w_v = dv[within_r, :].mean() if len(within_r) else 1e-6
    block = ((b_h / (w_h + 1e-6)) + (b_v / (w_v + 1e-6))) / 2.0
    return [float(b_h / 255.0), float(b_v / 255.0), float(block)], ["blk_h", "blk_v", "blk_ratio"]


def extract_features(path: Path):
    """All groups concatenated -> (np.float32 vector, names) or (None, None) on failure."""
    try:
        gray = load_gray(path)
        rgb = load_rgb(path)
        feats, names = [], []
        for fn in (feat_fft, feat_srm, feat_lbp, feat_blockiness):
            f, n = fn(gray); feats += f; names += n
        for fn in (feat_specular, feat_color):
            f, n = fn(rgb); feats += f; names += n
        arr = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return arr, names
    except Exception as e:                          # corrupt / unreadable image
        print(f"  skip {path.name}: {e}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
def list_images(d: Path):
    return sorted(p for p in d.glob("*") if p.suffix.lower() in EXTS) if d.exists() else []


def build_split(split: str):
    X, y, files, names, mps = [], [], [], None, []
    for label, cls in ((1, "positives"), (0, "negatives")):
        d = DATA / split / cls
        imgs = list_images(d)
        print(f"  {split}/{cls}: {len(imgs)} images")
        for p in imgs:
            arr, nm = extract_features(p)
            if arr is None:
                continue
            with Image.open(p) as im:
                mps.append(im.size[0] * im.size[1] / 1e6)
            X.append(arr); y.append(label); files.append(p)
            if names is None:
                names = nm
    if not X:
        raise RuntimeError(f"no images found under {DATA/split} — run split_dataset.py first")
    return np.array(X), np.array(y), files, names, np.array(mps)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def best_f1_threshold(y, scores):
    cand = np.unique(scores)
    best_t, best_f1 = 0.5, -1.0
    for t in cand:
        pred = (scores >= t).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0, p, r


def evaluate(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    TP = int(((y == 1) & (pred == 1)).sum())
    FP = int(((y == 0) & (pred == 1)).sum())
    TN = int(((y == 0) & (pred == 0)).sum())
    FN = int(((y == 1) & (pred == 0)).sum())
    f1_pos, prec, rec = _f1(TP, FP, FN)
    f1_neg, _, _ = _f1(TN, FN, FP)               # negatives as the "positive" -> symmetric
    total = TP + FP + TN + FN
    conf = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
    metrics = {
        "accuracy": (TP + TN) / total if total else 0.0,
        "precision_pos": prec,
        "recall_pos": rec,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "f1_macro": (f1_pos + f1_neg) / 2.0,
    }
    return conf, metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(SEED)
    MODELS.mkdir(parents=True, exist_ok=True)
    # fresh output
    for sub in ("positives", "negatives"):
        d = OUTPUT / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print("extracting TRAIN features ...")
    Xtr, ytr, ftr, names, mp_tr = build_split("train")
    print("extracting TEST features ...")
    Xte, yte, fte, _, mp_te = build_split("test")
    print(f"features/image: {Xtr.shape[1]}   train={len(ytr)}  test={len(yte)}")

    med_mp = float(np.median(np.concatenate([mp_tr, mp_te])))
    if med_mp < 1.0:
        print(f"WARNING: median resolution {med_mp:.2f} MP (<1 MP). Texture signal may be "
              f"degraded by compression - forensics ceiling is limited; fix the capture pipeline.")

    # carve a val slice from train for early stopping + threshold (NEVER touch test)
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.15, random_state=SEED, stratify=ytr)
    pos, neg = int((yt == 1).sum()), int((yt == 0).sum())
    dtr = lgb.Dataset(Xt, yt, feature_name=names)
    dvl = lgb.Dataset(Xv, yv, reference=dtr)
    params = dict(objective="binary", metric=["auc", "binary_logloss"],
                  num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=10,
                  scale_pos_weight=neg / max(pos, 1), verbose=-1, seed=SEED)
    print("training LightGBM ...")
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dvl],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    thr = best_f1_threshold(yv, booster.predict(Xv))      # threshold tuned on val, not test

    scores = booster.predict(Xte)
    pred = (scores >= thr).astype(int)
    conf, metrics = evaluate(yte, pred)

    # ── per-file json + sort test images by prediction ───────────────────────
    cls_name = {1: "positives", 0: "negatives"}
    per_file = []
    for p, tl, sc, pr in zip(fte, yte, scores, pred):
        shutil.copy2(p, OUTPUT / cls_name[int(pr)] / p.name)
        per_file.append({
            "file": p.name,
            "true_label": cls_name[int(tl)],
            "true": int(tl),
            "score_positive": round(float(sc), 6),   # P(asli); decision = score >= threshold
            "pred_label": cls_name[int(pr)],
            "pred": int(pr),
            "correct": bool(int(tl) == int(pr)),
        })

    results = {
        "class_map": {"positive(1)": "positives = KTP asli",
                      "negative(0)": "negatives = spoof (pendar/bezel)"},
        "threshold": round(thr, 6),
        "n_test": len(yte),
        "median_megapixels": round(med_mp, 3),
        "confusion": conf,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "clip_baseline_f1": CLIP_BASELINE_F1,
        "per_file": per_file,
    }
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2))

    # ── persist model ─────────────────────────────────────────────────────────
    booster.save_model(str(MODELS / "pad_lgbm.txt"), num_iteration=booster.best_iteration)
    (MODELS / "pad_meta.json").write_text(json.dumps({
        "threshold": thr, "feature_names": names,
        "class_map": results["class_map"], "max_side": MAX_SIDE,
    }, indent=2))

    # ── console summary ───────────────────────────────────────────────────────
    print("\n================ TEST RESULTS ================")
    print(f"threshold (val-tuned) : {thr:.4f}")
    print(f"TP {conf['TP']}  FP {conf['FP']}  TN {conf['TN']}  FN {conf['FN']}")
    print(f"accuracy      : {metrics['accuracy']*100:.2f}%")
    print(f"precision(pos): {metrics['precision_pos']*100:.2f}%")
    print(f"recall(pos)   : {metrics['recall_pos']*100:.2f}%")
    print(f"F1 (positives): {metrics['f1_pos']*100:.2f}%")
    print(f"F1 (negatives): {metrics['f1_neg']*100:.2f}%")
    print(f"F1 (macro)    : {metrics['f1_macro']*100:.2f}%   <-- compare vs CLIP {CLIP_BASELINE_F1*100:.0f}%")
    delta = metrics["f1_macro"] - CLIP_BASELINE_F1
    print(f"vs CLIP baseline: {'+' if delta>=0 else ''}{delta*100:.1f} pts")
    print("top-15 features:")
    imp = sorted(zip(names, booster.feature_importance(importance_type="gain")),
                 key=lambda t: -t[1])[:15]
    for nm, g in imp:
        print(f"   {nm:16s} {g:.0f}")
    print(f"\nsaved: {MODELS/'pad_lgbm.txt'} , {OUTPUT/'results.json'} , {OUTPUT}/(positives|negatives)/")


if __name__ == "__main__":
    main()
