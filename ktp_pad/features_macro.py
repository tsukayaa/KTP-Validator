"""
CODE 2b — MACRO-cue features: card-anchored surround, dark region, over-crop.

READ THIS FIRST — the dk_* group did not measure what it was designed to measure.
On the real set (1171 genuine / 1372 spoof) the mask overlays showed
`largest_border_blob` almost never locks onto the monitor or the shadow. On genuine
it grabs the wall / floor / hoodie / bed; on spoof it is a mix of monitor, hand,
shadow and desk. It finds "the largest dark thing touching the frame", which is
usually just the room.

So the dk_* AUCs in output/macro_diag.md are NOT evidence about screen-vs-shadow;
they are room-background statistics, and the physics hypothesis behind dk_* remains
UNTESTED rather than refuted. Worse, the strongest of them (dk_v_p50, AUC 0.218 =
"spoof surroundings are darker") is a plausible ACQUISITION-ENVIRONMENT shortcut:
people who photograph a screen sit at a desk in a dim room, people who photograph a
real card use a bed or a floor. That collapses across users and in production.

The fix is the cd_*/sr_* group below. The data pointed at it: oc_ring_chi2 (0.740)
and oc_ring_v_ratio (0.245 ≡ 0.755) were among the strongest features and use NO
blob detection at all -- just an annulus. Anchoring that annulus to the card instead
of to the image frame is what sr_* does. dk_* is retained unchanged so the next
diag_macro run is a direct A/B, not a guess.

KNOWN LIMITATION of find_card, measured on synthetic scenes: when the card fills the
whole frame (a true over-crop, no background at all) Otsu has no card-vs-background
split to find, so it splits print-vs-blank-margin instead and the box comes back as
the PRINT extent -- ~40 px short of the frame on each side. The features still move
hard in the right direction (cd_margin_min 0.061 vs 0.211, cd_area 0.71 vs 0.32), and
over-crop is a weak feature by design, so this is accepted rather than chased. It does
mean cd_margin_* must not be read as a calibrated distance.

WHY THIS IS A SEPARATE MODULE FROM train_pad.py:

1. train_pad.py's loaders CENTER-CROP to 1024 px. That is correct for the texture
   features (resizing destroys moire/SRM signal, so we crop instead) but it throws
   away the image border -- which is exactly where the evidence these features look
   for lives. So everything here runs on the FULL frame, downscaled to 512. Resizing
   is SAFE here and forbidden there: these are geometry/layout/colour statistics, not
   high-frequency texture.

2. The dark-region problem is NOT "is there a black blob". Both classes have one:
      spoof   -> the monitor/screen area that was never cropped out of the shot
      genuine -> the shadow cast by the person taking the photo
   A naive "dark blob => spoof" feature is worse than useless, it actively costs
   BPCER. So we do not emit "how dark" or "how big" as the decisive signal. We emit
   the properties that physically separate a cast shadow from a display surface:

      edge       shadow has a penumbra (gradual);  a screen edge is a step
      shape      shadow boundary curves (body/hand outline);  screen boundary is a line
      interior   shadow still contains the underlying surface texture, merely
                 attenuated -- normalise the gain and it reappears.  A dark screen
                 region contains nothing but sensor noise.
      colour     shadow = same material, so its mean RGB vector stays roughly
                 COLLINEAR with the lit region's (just shorter).  A panel's black is
                 neutral / blue-cast and points somewhere else.
      ramp       shadow carries a light falloff;  screen black is flat.
      position   a shadow can fall ACROSS the card;  the screen area around a
                 displayed card never covers the card itself.

   The load-bearing four are `dk_struct` (texture vs noise), `dk_rgb_cos` (colour
   collinearity), `dk_edge_rise` / `dk_penumbra` (edge softness) and `dk_inner_frac`
   (does it cross the card). All four are level-independent, which matters because
   the black level varies wildly between panels and exposures.
   `dk_detail_ratio_gain` expresses the same idea as `dk_struct` but through
   luminance normalisation, and on the synthetic check it was unstable and even
   pointed the wrong way -- it is kept as a candidate, not relied on. The
   size/darkness features are context only, so the tree can condition on them; do
   NOT expect them to separate the classes on their own.

3. Over-crop ("KTP di-zoom sampai tidak ada background") is deliberately a WEAK
   feature here, not a gate. Honest users crop tightly too; treating over-crop as
   evidence of spoof would reject real customers. It is emitted so the tree can use
   it as a mild prior -- a fraudster cropping away the monitor produces an
   over-cropped frame, so it partially covers the case the dk_* features lose.

No new dependencies: numpy + scipy + Pillow only (same as train_pad.py). No cv2.

Every feature is finite and defined even when no dark region exists (`dk_present`=0
gates the rest at 0.0) so LightGBM sees a rectangular matrix with no NaNs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import kurtosis

MACRO_SIDE = 512          # full frame downscaled to this; geometry survives, texture need not
KTP_ASPECT = 85.6 / 54.0  # 1.585 — ISO/IEC 7810 ID-1, the physical card ratio
_EPS = 1e-6
_VFLOOR = 0.05            # luminance floor for gain normalisation (see feat_dark)

# KV square high-pass — same kernel as SRM_KERNELS[4] in train_pad.py, duplicated
# here (not imported) to keep this module free of a circular import.
_KV = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0


def _local_std(V: np.ndarray, size: int = 5) -> np.ndarray:
    """Local standard deviation, numerically stable.

    The textbook form `uniform_filter(V**2) - uniform_filter(V)**2` is catastrophic
    cancellation waiting to happen: over a flat dark strip both terms are ~2.5e-3 and
    the true variance is ~1e-5, which is below float32's resolution at that magnitude,
    so the result collapses to ~0 or goes negative. That made the struct ratios below
    divide by nothing and blow up to ~100 on exactly the flat regions they are meant
    to score LOW. Subtracting the local mean first keeps the subtraction small.
    """
    V = V.astype(np.float64)
    mu = ndimage.uniform_filter(V, size)
    return np.sqrt(np.clip(ndimage.uniform_filter((V - mu) ** 2, size), 0, None))


def _core(region: np.ndarray, min_px: int = 64, erode: int = 4) -> np.ndarray:
    """Region minus a 4 px rim, for INTERIOR statistics.

    A 5x5 local window sitting on the outermost line of a strip straddles the
    boundary, so its local std is the size of the STEP, not of the strip's own
    texture. Left uncorrected that made every interior statistic a second, noisier
    copy of the edge-gradient feature.

    4 px, not the window radius of 2: _local_std is a TWO-pass estimator, so the
    contamination travels twice. `(V - mu)**2` is already wrong wherever mu is a
    mixed local mean (2 px in from the step), and the second uniform_filter then
    smears those wrong values another 2 px inward. Measured on a synthetic bar:
    eroding 2 px still left p95 of the local std at 26x the p50.

    Falls back to a smaller rim, then to the whole region, when eroding would leave
    too little to measure -- so on a very thin bar these interior stats are still
    edge-contaminated. fb_extent / fb_edge_jitter / fb_inner_grad do not use _core
    and stay trustworthy there.
    """
    for it in (erode, erode // 2):
        if it >= 1:
            core = ndimage.binary_erosion(region, np.ones((3, 3), bool), iterations=it)
            if int(core.sum()) >= min_px:
                return core
    return region


def _struct_ratio(loc: np.ndarray) -> float:
    """p95/p50 of local std: high for print/texture, ~1 for a flat panel or pure noise.

    The denominator is floored at one 8-bit quantisation step — below that the input
    carries no information anyway, and an unfloored ratio just amplifies rounding.
    """
    return float(np.percentile(loc, 95)) / max(float(np.percentile(loc, 50)), 1.0 / 255.0)


# ══════════════════════════════════════════════════════════════════════════════
# LOADING  (full frame, downscaled — the opposite of train_pad's center crop)
# ══════════════════════════════════════════════════════════════════════════════
def load_macro(path: Path) -> np.ndarray:
    """Whole image as float32 RGB, longest side <= MACRO_SIDE. Aspect preserved."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = MACRO_SIDE / max(w, h)
        if s < 1.0:
            im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BOX)
        return np.asarray(im, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DARK-REGION DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def dark_mask(V: np.ndarray) -> np.ndarray:
    """Dark pixels, thresholded RELATIVE to the bright content (not an absolute cut).

    A KTP shot under weak ambient light is globally dim; an absolute threshold would
    flag the whole frame. Anchoring to the 90th percentile keeps the threshold at
    'much darker than the card' regardless of overall exposure.
    """
    hi = float(np.percentile(V, 90))
    t = float(np.clip(0.45 * hi, 0.08, 0.45))
    m = V < t
    return ndimage.binary_opening(m, structure=np.ones((3, 3), bool), iterations=2)


def largest_border_blob(m: np.ndarray, min_frac: float = 0.015) -> np.ndarray | None:
    """Biggest dark component that touches the frame edge, or None.

    Border-touching is a *detection* filter, not a discriminator: both a monitor
    area and a photographer's shadow reach the edge. It only rejects interior dark
    stuff (the KTP photo box, printed black text) which is neither.
    """
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    border = np.zeros(m.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    for idx in np.argsort(-sizes):
        if sizes[idx] < min_frac * m.size:
            break                                    # sorted desc -> rest are smaller too
        c = lab == (idx + 1)
        if (c & border).any():
            return c
    return None


# ══════════════════════════════════════════════════════════════════════════════
# DARK-REGION FEATURES  (shadow vs display surface)
# ══════════════════════════════════════════════════════════════════════════════
DK_NAMES = [
    "dk_present",                                                    # gate
    "dk_area", "dk_bbox_fill", "dk_nsides", "dk_inner_frac", "dk_center_dark",
    "dk_edge_rms", "dk_edge_linear", "dk_edge_inlier",               # straightness
    "dk_edge_grad", "dk_penumbra", "dk_edge_rise",                   # sharpness
    "dk_detail_abs", "dk_detail_ratio", "dk_detail_ratio_gain", "dk_localstd",
    "dk_struct", "dk_res_kurt",                                      # structure vs noise
    "dk_rgb_cos", "dk_chroma", "dk_bluecast",                        # colour
    "dk_v_p50", "dk_v_p05", "dk_v_std", "dk_ramp", "dk_flat_resid",  # luminance
]


def feat_dark(rgb: np.ndarray):
    """23 features describing the dominant border-touching dark region (if any)."""
    f = dict.fromkeys(DK_NAMES, 0.0)
    h, w = rgb.shape[:2]
    V = rgb.mean(axis=2) / 255.0
    c = largest_border_blob(dark_mask(V))
    if c is None or int(c.sum()) < 50:
        return [f[k] for k in DK_NAMES], list(DK_NAMES)

    f["dk_present"] = 1.0
    npix = float(c.sum())
    f["dk_area"] = npix / c.size

    ys, xs = np.nonzero(c)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    f["dk_bbox_fill"] = npix / float((y1 - y0 + 1) * (x1 - x0 + 1))   # rectangle -> ~1
    f["dk_nsides"] = float(c[0].any() + c[-1].any() + c[:, 0].any() + c[:, -1].any())

    # a shadow can lie across the card; the screen area around a displayed card cannot
    inner = np.zeros(c.shape, bool)
    inner[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)] = True
    f["dk_inner_frac"] = float((c & inner).sum()) / npix
    f["dk_center_dark"] = float((c & inner).sum()) / float(max(inner.sum(), 1))

    # ── boundary: straight line (screen) vs curved outline (shadow) ────────────
    bnd = c & ~ndimage.binary_erosion(c, np.ones((3, 3), bool))
    bnd[0, :] = bnd[-1, :] = False                    # frame edge is not a real boundary
    bnd[:, 0] = bnd[:, -1] = False
    by, bx = np.nonzero(bnd)
    if len(by) >= 20:
        pts = np.stack([bx, by], axis=1).astype(np.float64)
        d = pts - pts.mean(axis=0)
        ev, evec = np.linalg.eigh(d.T @ d / len(d))   # ascending eigenvalues
        diag = float(np.hypot(h, w))
        rms = float(np.sqrt(max(ev[0], 0.0)))
        f["dk_edge_rms"] = rms / diag
        f["dk_edge_linear"] = 1.0 - ev[0] / (ev[1] + _EPS)
        dist = np.abs(d @ evec[:, 0])                 # distance to the total-least-squares line
        f["dk_edge_inlier"] = float((dist < 1.5).mean())
        gy, gx = np.gradient(V)
        f["dk_edge_grad"] = float(np.hypot(gx[bnd], gy[bnd]).mean())

    # ── penumbra: how much of the transition band sits at intermediate luminance ─
    st = np.ones((3, 3), bool)
    dil = ndimage.binary_dilation(c, st, iterations=5)
    ero = ndimage.binary_erosion(c, st, iterations=5)
    band = dil & ~ero
    band[0, :] = band[-1, :] = False
    band[:, 0] = band[:, -1] = False
    lit = ~dil
    if band.any() and ero.any() and lit.any():
        v_dark = float(np.percentile(V[ero], 90))
        v_lit = float(np.percentile(V[lit], 10))
        if v_lit > v_dark:
            mid = band & (V > v_dark) & (V < v_lit)
            f["dk_penumbra"] = float(mid.sum()) / float(band.sum())
            # estimated 0-100% rise distance in px: contrast / slope at the boundary.
            # step edge -> 1-2 px, penumbra -> tens of px.
            if f["dk_edge_grad"] > 0:
                f["dk_edge_rise"] = (v_lit - v_dark) / f["dk_edge_grad"]

    # ── interior: attenuated real texture vs nothing but a noise floor ─────────
    res = ndimage.convolve(V, _KV, mode="reflect")
    if ero.any() and lit.any():
        sd, sl = float(res[ero].std()), float(res[lit].std())
        md, ml = float(V[ero].mean()), float(V[lit].mean())
        f["dk_detail_abs"] = sd
        f["dk_detail_ratio"] = sd / (sl + _EPS)
        # Gain-normalised, WITH A FLOOR. A shadow attenuates the scene multiplicatively
        # so contrast/mean is roughly preserved; a dark panel has no scene content at
        # all. Without _VFLOOR the panel's near-zero mean makes this blow UP instead of
        # down. NOTE: even floored, this stayed ambiguous on the synthetic check
        # (screen 2.78 vs shadow 2.39 — the wrong way round). Kept as a candidate;
        # dk_struct below is the level-independent version and is the one to trust.
        f["dk_detail_ratio_gain"] = ((sd / (md + _VFLOOR)) /
                                     ((sl / (ml + _VFLOOR)) + _EPS))
        ld = _local_std(V)[ero]
        f["dk_localstd"] = float(ld.mean())
        # Structure vs white noise — LEVEL-INDEPENDENT, so it survives whatever the
        # black level happens to be. Real texture is spatially clustered (flat areas
        # plus sharp edges) -> tall p95/p50 and heavy-tailed residual. Sensor noise is
        # uniform -> p95/p50 ~1.3, kurtosis ~0.
        f["dk_struct"] = _struct_ratio(ld)
        rd = res[ero]
        f["dk_res_kurt"] = float(kurtosis(rd, fisher=True, bias=False)) if rd.std() > _EPS else 0.0

        # ── colour: same material stays collinear in RGB, a panel does not ─────
        md_rgb = rgb[ero].mean(axis=0)
        ml_rgb = rgb[lit].mean(axis=0)
        f["dk_rgb_cos"] = float(md_rgb @ ml_rgb /
                                (np.linalg.norm(md_rgb) * np.linalg.norm(ml_rgb) + _EPS))
        f["dk_chroma"] = float((md_rgb.max() - md_rgb.min()) / (md_rgb.max() + _EPS))
        f["dk_bluecast"] = float((md_rgb[2] - md_rgb[0]) / (md_rgb.mean() + _EPS))

    # ── luminance: level, spread, and light falloff ────────────────────────────
    vd = V[c]
    f["dk_v_p50"] = float(np.percentile(vd, 50))
    f["dk_v_p05"] = float(np.percentile(vd, 5))
    f["dk_v_std"] = float(vd.std())
    A = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64)
    sol, *_ = np.linalg.lstsq(A, vd.astype(np.float64), rcond=None)
    f["dk_ramp"] = float(np.hypot(sol[0], sol[1]) * max(h, w))   # V change across the frame
    f["dk_flat_resid"] = float((vd - A @ sol).std())             # flatness AFTER removing the ramp
    return [f[k] for k in DK_NAMES], list(DK_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# OVER-CROP FEATURES  (weak prior only — see module docstring point 3)
# ══════════════════════════════════════════════════════════════════════════════
def _edge_margin(prof: np.ndarray, span: float = 0.30, min_prom: float = 2.0):
    """Outermost strong gradient peak within `span` of an edge -> (margin, prominence).

    The card's own border is a strong localised ridge in the gradient profile. No
    ridge near the frame edge => no visible card border => nothing but card in shot.
    Returns (span, 0.0) when nothing prominent is found, so 'no edge' reads as the
    maximum searched margin rather than a missing value.
    """
    n = len(prof)
    if n == 0:                                   # degenerate frame (a 1 px side)
        return span, 0.0
    k = max(3, int(n * span))
    base = float(np.median(prof)) + _EPS
    lead = prof[:k]
    i = int(np.argmax(lead))
    prom = float(lead[i]) / base
    if prom < min_prom:
        return span, 0.0
    return i / float(n), prom


OC_NAMES = [
    "oc_margin_l", "oc_margin_r", "oc_margin_t", "oc_margin_b",
    "oc_margin_min", "oc_margin_mean", "oc_edges_found",
    "oc_prom_mean",
    "oc_ring_chi2", "oc_ring_sat", "oc_ring_sat_ratio", "oc_ring_v_ratio",
    "oc_aspect", "oc_aspect_dev",
]


def feat_overcrop(rgb: np.ndarray):
    """14 features on 'is there any background left around the card'."""
    h, w = rgb.shape[:2]
    V = rgb.mean(axis=2) / 255.0

    # gradient profiles: vertical card edges show up as ridges in the column profile
    gx = np.abs(np.diff(V, axis=1)).mean(axis=0)
    gy = np.abs(np.diff(V, axis=0)).mean(axis=1)
    ml, pl = _edge_margin(gx)
    mr, pr = _edge_margin(gx[::-1])
    mt, pt = _edge_margin(gy)
    mb, pb = _edge_margin(gy[::-1])
    margins = [ml, mr, mt, mb]
    proms = [pl, pr, pt, pb]

    # ring vs centre: with background present the two differ; on an over-crop the
    # ring is card as well, so the distributions collapse onto each other
    b = max(2, int(0.05 * min(h, w)))
    ring = np.zeros((h, w), bool)
    ring[:b, :] = ring[-b:, :] = True
    ring[:, :b] = ring[:, -b:] = True
    ctr = np.zeros((h, w), bool)
    ctr[int(0.25 * h):int(0.75 * h), int(0.25 * w):int(0.75 * w)] = True
    for sel in (ring, ctr):                      # a tiny frame can leave either empty
        if not sel.any():
            sel[:] = True

    chi2 = 0.0
    for ci in range(3):
        a, _ = np.histogram(rgb[:, :, ci][ring], bins=32, range=(0, 255), density=True)
        c_, _ = np.histogram(rgb[:, :, ci][ctr], bins=32, range=(0, 255), density=True)
        chi2 += 0.5 * float(((a - c_) ** 2 / (a + c_ + _EPS)).sum())
    chi2 /= 3.0

    mx = rgb.max(axis=2)
    S = np.where(mx > 0, (mx - rgb.min(axis=2)) / (mx + _EPS), 0.0)
    s_ring, s_ctr = float(S[ring].mean()), float(S[ctr].mean())
    v_ring, v_ctr = float(V[ring].mean()), float(V[ctr].mean())

    asp = max(w, h) / float(min(w, h))
    feats = margins + [
        float(min(margins)), float(np.mean(margins)),
        float(sum(p > 0 for p in proms)), float(np.mean(proms)),
        chi2, s_ring, s_ring / (s_ctr + _EPS), v_ring / (v_ctr + _EPS),
        asp, abs(asp - KTP_ASPECT),
    ]
    return feats, list(OC_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# CARD LOCALISATION + CARD-ANCHORED SURROUND   (the dk_* replacement)
# ══════════════════════════════════════════════════════════════════════════════
def _otsu(x: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold. Used instead of a percentile because a percentile selects a
    FIXED FRACTION of the image by construction: on an over-cropped frame the card is
    ~90% of the pixels, so a p50 anchor lands inside the card, the threshold comes out
    far too high and the card shatters into fragments. Otsu finds the actual split."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return hi
    hist, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2.0
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    sigma_b[~np.isfinite(sigma_b)] = 0.0
    return float(centres[int(np.argmax(sigma_b))])


def _refine_box(V, C, box, frac=0.30, min_step=0.035):
    """Snap each side of the box to the card boundary, if a boundary is visible there.

    Uses the larger of the LUMINANCE step and the CHROMA step. Luminance alone misses
    a KTP lying on a white desk or a sheet of paper -- the card's mean brightness is
    close to the paper's, so there is no visible step -- but the card is blue and the
    paper is neutral, so the chroma step is large.

    The smoothed energy blob usually already reaches the card edge, but not always --
    a KTP has a blank margin around the print, and when that margin is wide the box
    lands inside the card, which would put the surround strips on the card's own
    border instead of on the background. So scan outward from each side and take the
    largest luminance step; that step is the card edge.

    A side with no clear step is LEFT ALONE. An earlier version padded outward in
    that case, which was backwards: "no step in the window" almost always means the
    box is already at or past the card edge, and padding then pushed it ~30px into
    the background on every side.
    """
    h, w = V.shape
    y0, y1, x0, x1 = box
    bh, bw = y1 - y0 + 1, x1 - x0 + 1

    def step(sl, axis):
        """Largest |difference| along `axis` of the mean profile over `sl`, luma or chroma."""
        dv = np.abs(np.diff(V[sl].mean(axis=axis)))
        dc = np.abs(np.diff(C[sl].mean(axis=axis)))
        d = np.maximum(dv, dc)
        return (int(np.argmax(d)), float(d.max())) if d.size else (0, 0.0)

    a = max(0, x0 - max(1, int(frac * bw)))
    if x0 - a >= 2:
        i, s = step((slice(y0, y1 + 1), slice(a, x0 + 1)), 0)
        if s >= min_step:
            x0 = a + i + 1
    b = min(w - 1, x1 + max(1, int(frac * bw)))
    if b - x1 >= 2:
        i, s = step((slice(y0, y1 + 1), slice(x1, b + 1)), 0)
        if s >= min_step:
            x1 = x1 + i
    a = max(0, y0 - max(1, int(frac * bh)))
    if y0 - a >= 2:
        i, s = step((slice(a, y0 + 1), slice(x0, x1 + 1)), 1)
        if s >= min_step:
            y0 = a + i + 1
    b = min(h - 1, y1 + max(1, int(frac * bh)))
    if b - y1 >= 2:
        i, s = step((slice(y1, b + 1), slice(x0, x1 + 1)), 1)
        if s >= min_step:
            y1 = y1 + i
    return int(y0), int(y1), int(x0), int(x1)


_CARD_AREA = (0.06, 0.72)     # plausible card box as a fraction of the frame
_CARD_ASPECT_MAX = 2.30       # KTP is 1.585; allows perspective, rejects frame-shaped boxes
_CARD_ASPECT_GOOD = 0.40      # |aspect - KTP_ASPECT| below this = stop raising the threshold


def _box_from(m: np.ndarray):
    """Largest filled component of `m` -> ((y0,y1,x0,x1), fill ratio) or None.

    `fill` is the component's own area over its bounding box. A card photographed
    roughly square-on fills its box; a box that has swallowed several unrelated
    background regions does not. It is the only self-check available without labels.
    """
    m = ndimage.binary_closing(m, np.ones((5, 5), bool), iterations=2)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    c = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(c)
    box = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    if box[1] - box[0] < 8 or box[3] - box[2] < 8:
        return None
    area = (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
    return box, float(c.sum()) / float(area)


def find_card(rgb: np.ndarray):
    """Locate the KTP by PRINT STRUCTURE, not by colour.
    -> (mask, (y0,y1,x0,x1), fill, ok) or None.

    Colour-based segmentation fails on this data: KTP blue varies with print run,
    fading, white-balance and photocopies, and a displayed card takes the panel's
    gamut. Structure does not care -- and it works identically on a real card and a
    displayed one, which is the point: this step must NOT itself discriminate the
    classes, it only has to find the card so the surround can be measured.

    Two things a naive "local gradient energy" version got wrong, both fixed here:

    * Textured backgrounds (bedsheet, hoodie, carpet, wood grain) are high-gradient
      too, so the box swallowed the whole frame. Fixed by smoothing first: sensor
      noise and fabric grain live at the pixel scale, printed strokes are several
      pixels wide at 512, so a small Gaussian removes the former and keeps the latter.

    * A percentile threshold selects a FIXED FRACTION of the image by construction,
      which is structurally wrong -- an over-cropped frame is nearly all card, so a
      p50 anchor sits inside the card and the threshold comes out far too high.
      Replaced with Otsu (see _otsu).

    A SINGLE Otsu threshold was still not enough, which the real-data overlays showed:
    of ten sampled images only three boxes landed on the card. Two swallowed the whole
    frame -- and the reviewer confirmed those frames had a WIDE visible background, so
    this is not the "card fills the frame" limitation, it is Otsu cutting too low and
    letting background texture join the card component. Three more came back roughly
    1.4x the card. Meanwhile `cd_found` reported 99% on both classes, because it only
    ever meant "the function returned something".

    So: sweep the threshold upward from Otsu and keep the first box that is actually
    card-SHAPED. Raising the cut peels background texture off the component before it
    touches the print, which is denser. The plausibility test is the ID-1 aspect ratio
    plus an area range -- both are properties of a card, not of a class, so this stays
    class-neutral. When nothing plausible turns up the best candidate is still returned
    with ok=0 rather than dropped, so `cd_ok` can gate honestly instead of hiding the
    failure behind a None.
    """
    h, w = rgb.shape[:2]
    V = rgb.mean(axis=2) / 255.0
    C = (rgb.max(axis=2) - rgb.min(axis=2)) / 255.0    # chroma, for the white-desk case
    gy, gx = np.gradient(ndimage.gaussian_filter(V, 1.2))
    k = max(5, int(0.025 * min(h, w)))
    E = ndimage.uniform_filter(np.hypot(gx, gy), k)
    if float(E.max() - E.min()) < 1e-6:
        return None                                   # featureless frame, nothing to find

    t0, hi = _otsu(E), float(E.max())
    best = None
    for step in (0.0, 0.15, 0.30, 0.45, 0.60):
        got = _box_from(E > t0 + step * (hi - t0))
        if got is None:
            continue
        y0, y1, x0, x1 = _refine_box(V, C, got[0])
        bh, bw = y1 - y0 + 1, x1 - x0 + 1
        if bh < 8 or bw < 8:
            continue
        area = bh * bw / float(h * w)
        asp = max(bh, bw) / float(min(bh, bw))
        dev = abs(asp - KTP_ASPECT)
        ok = _CARD_AREA[0] <= area <= _CARD_AREA[1] and asp <= _CARD_ASPECT_MAX
        # A plausible box always beats an implausible one, whatever their aspects.
        score = dev + (0.0 if ok else 100.0)
        if best is None or score < best[0]:
            best = (score, (y0, y1, x0, x1), got[1], ok)
        if ok and dev < _CARD_ASPECT_GOOD:
            break                                     # good enough; stop peeling
    if best is None:
        return None
    _, (y0, y1, x0, x1), fill, ok = best
    mask = np.zeros(V.shape, bool)
    mask[y0:y1 + 1, x0:x1 + 1] = True     # card = the refined rectangle, blank margin included
    return mask, (y0, y1, x0, x1), float(fill), bool(ok)


def _slice_area(s) -> int:
    return max(0, s[0].stop - s[0].start) * max(0, s[1].stop - s[1].start)


def _strips(shape, box, pad):
    """The four bands just OUTSIDE the card box — where an uncropped screen lives."""
    h, w = shape
    y0, y1, x0, x1 = box
    out = {}
    if x0 > 0:
        out["l"] = (slice(y0, y1 + 1), slice(max(0, x0 - pad), x0))
    if x1 < w - 1:
        out["r"] = (slice(y0, y1 + 1), slice(x1 + 1, min(w, x1 + 1 + pad)))
    if y0 > 0:
        out["t"] = (slice(max(0, y0 - pad), y0), slice(x0, x1 + 1))
    if y1 < h - 1:
        out["b"] = (slice(y1 + 1, min(h, y1 + 1 + pad)), slice(x0, x1 + 1))
    return {k: s for k, s in out.items()
            if _slice_area(s) >= 64}                  # ignore slivers — stats would be noise


CD_NAMES = [
    "cd_found", "cd_ok", "cd_area", "cd_fill", "cd_aspect", "cd_aspect_dev",
    "cd_margin_l", "cd_margin_r", "cd_margin_t", "cd_margin_b",
    "cd_margin_min", "cd_margin_mean", "cd_nsides_open",
]
SR_NAMES = [
    "sr_nstrips", "sr_v_mean", "sr_v_std",
    "sr_side_vmin", "sr_side_vmax", "sr_side_contrast",
    "sr_dk_v", "sr_dk_flat", "sr_dk_struct", "sr_dk_detail", "sr_dk_kurt",
    "sr_dk_chroma", "sr_dk_bluecast", "sr_dk_rgbcos",
    "sr_card_contrast", "sr_edge_grad",
]


def feat_card(rgb: np.ndarray):
    """Card box + the annulus around it. 28 features.

    Every sr_* stat is taken on the SINGLE DARKEST of the four surround strips, not
    on their average. An uncropped monitor shows up on one or two sides only, so
    averaging dilutes it away; the min picks the strip most likely to BE the screen.
    `sr_side_contrast` carries the asymmetry itself — a uniform desk or wall gives a
    small spread across the four sides, a screen edge on one side gives a large one.
    """
    f = dict.fromkeys(CD_NAMES + SR_NAMES, 0.0)
    names = CD_NAMES + SR_NAMES
    h, w = rgb.shape[:2]
    V = rgb.mean(axis=2) / 255.0

    got = find_card(rgb)
    if got is None:
        return [f[k] for k in names], list(names)
    cmask, (y0, y1, x0, x1), fill, ok = got

    f["cd_found"] = 1.0     # "the function returned a box" — NOT a quality claim
    f["cd_ok"] = float(ok)  # "the box is card-SHAPED" — the gate to actually trust
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    f["cd_area"] = bh * bw / float(h * w)
    # Was `cmask.sum() / (bh*bw)` and therefore identically 1.0 for every image, since
    # cmask IS the filled rectangle. Dead by construction — it showed up in the real run
    # as AUC 0.496 in all four columns, byte-identical to cd_found and sr_nstrips, which
    # is the signature of a constant feature. Now the component's own fill of its box.
    f["cd_fill"] = fill
    asp = max(bh, bw) / float(min(bh, bw))
    f["cd_aspect"] = asp
    f["cd_aspect_dev"] = abs(asp - KTP_ASPECT)

    # true over-crop measure: distance from the card box to each frame edge
    mg = [x0 / float(w), (w - 1 - x1) / float(w), y0 / float(h), (h - 1 - y1) / float(h)]
    f["cd_margin_l"], f["cd_margin_r"], f["cd_margin_t"], f["cd_margin_b"] = mg
    f["cd_margin_min"] = float(min(mg))
    f["cd_margin_mean"] = float(np.mean(mg))
    f["cd_nsides_open"] = float(sum(m > 0.01 for m in mg))

    pad = max(4, int(0.06 * max(h, w)))
    st = _strips((h, w), (y0, y1, x0, x1), pad)
    f["sr_nstrips"] = float(len(st))
    if not st:
        return [f[k] for k in names], list(names)     # card fills the frame: no surround

    means = {k: float(V[s].mean()) for k, s in st.items()}
    allv = np.concatenate([V[s].ravel() for s in st.values()])
    f["sr_v_mean"] = float(allv.mean())
    f["sr_v_std"] = float(allv.std())
    f["sr_side_vmin"] = float(min(means.values()))
    f["sr_side_vmax"] = float(max(means.values()))
    f["sr_side_contrast"] = f["sr_side_vmax"] - f["sr_side_vmin"]

    # the darkest strip — interior only: it abuts the card on one edge and the wider
    # background on the other, so an uneroded window would straddle both (see _core)
    dk = _core(_mask_of(st[min(means, key=means.get)], (h, w)))
    vd = V[dk]
    f["sr_dk_v"] = float(vd.mean())

    ys, xs = np.nonzero(dk)
    A = np.stack([xs, ys, np.ones(xs.size)], axis=1).astype(np.float64)
    sol, *_ = np.linalg.lstsq(A, vd.astype(np.float64), rcond=None)
    f["sr_dk_flat"] = float((vd - A @ sol).std())      # flat AFTER removing any ramp

    res = ndimage.convolve(V, _KV, mode="reflect")[dk]
    f["sr_dk_detail"] = float(res.std())
    f["sr_dk_kurt"] = float(kurtosis(res, fisher=True, bias=False)) if res.std() > _EPS else 0.0
    f["sr_dk_struct"] = _struct_ratio(_local_std(V)[dk])

    md = rgb[dk].reshape(-1, 3).mean(axis=0)
    mc = rgb[cmask].mean(axis=0)
    f["sr_dk_chroma"] = float((md.max() - md.min()) / (md.max() + _EPS))
    f["sr_dk_bluecast"] = float((md[2] - md[0]) / (md.mean() + _EPS))
    f["sr_dk_rgbcos"] = float(md @ mc / (np.linalg.norm(md) * np.linalg.norm(mc) + _EPS))
    f["sr_card_contrast"] = float(V[cmask].mean()) / (f["sr_dk_v"] + _VFLOOR)

    # gradient right at the card border: a card rendered on a panel has a crisp
    # digital edge, a physical card sitting on a surface has a soft/shadowed one
    box = np.zeros((h, w), bool)
    box[y0:y1 + 1, x0:x1 + 1] = True
    bnd = ndimage.binary_dilation(box, np.ones((3, 3), bool)) & ~box
    if bnd.any():
        gy2, gx2 = np.gradient(V)
        f["sr_edge_grad"] = float(np.hypot(gx2[bnd], gy2[bnd]).mean())

    return [f[k] for k in names], list(names)


# ══════════════════════════════════════════════════════════════════════════════
# FRAME-BORDER RUNS  (the pillarbox case — see the note below)
# ══════════════════════════════════════════════════════════════════════════════
# A real example from the data: the KTP sits in the MIDDLE of a bright scene, and the
# uncropped monitor shows up only as thin black pillarbox strips along the far left
# and right of the FRAME. Neither of the other groups sees it:
#
#   sr_*  hugs the card, so its strips land on the bright background between the card
#         and the bars, and miss them entirely.
#   oc_*  averages a ring over all four sides, so a bar on one side is diluted.
#   dk_*  may or may not pick it up, depending on what else is dark in the room.
#
# So scan inward from each FRAME edge for a contiguous run of columns/rows that are
# mostly dark. The discriminator against a photographer's shadow is not darkness but
# GEOMETRY: a pillarbox bar spans the full height at a constant width (fb_edge_jitter
# ~0, fb_extent ~1), while a shadow reaching the frame edge has a ragged, varying
# boundary. The same shadow-vs-panel physics as sr_* is then applied to the run.
FB_NAMES = [
    "fb_run_l", "fb_run_r", "fb_run_t", "fb_run_b", "fb_run_max",
    "fb_nsides", "fb_sym_lr", "fb_sym_tb",
    "fb_extent", "fb_edge_jitter", "fb_inner_grad",
    "fb_v", "fb_contrast", "fb_flat", "fb_struct", "fb_detail", "fb_kurt",
    "fb_chroma", "fb_bluecast", "fb_rgbcos",
]
_FB_COVER = 0.75          # a line counts as "in the run" if this fraction of it is dark
_FB_DARK = 0.22           # "dark" ceiling — see _dark_cut. One knob, tune here.


def _dark_cut(V: np.ndarray) -> float:
    """Threshold for "as dark as an unlit panel", anchored to the bright content.

    The first version was `clip(0.45 * p90, 0.08, 0.45)` and the real-data run showed
    exactly how wrong that is. On a well-lit photo p90 is ~0.9, so the cut lands near
    0.40 -- MID-GREY. A wooden desk, a concrete floor, dark jeans all fall below it.
    The frame-run gate fired on 37.3% of genuine images, and the reviewer's verdict on
    those overlays was that the runs sat on ordinary background that was "not even a
    dark colour".

    The AUC table said the same thing independently: with labels 1 = spoof, `fb_v`
    inside its own subpopulation came out 0.176, i.e. the runs found on genuine frames
    are markedly BRIGHTER than those on spoof frames. That is not a discovery about
    genuine images, it is the loose threshold showing up as apparent signal.

    An unlit LCD sits around V 0.05-0.12; 0.22 keeps that with headroom and excludes
    mid-tones. The p90 anchor is kept so an underexposed frame still scales down, but
    it can now only ever LOWER the cut, never raise it above _FB_DARK.

    Falsifiable prediction of this change: the genuine `fb_run_max>0` rate should fall
    from 37.3% to well under 15% while spoof stays above 70%. If spoof falls too, the
    bars are not as dark as assumed and _FB_DARK is the wrong knob.
    """
    return float(np.clip(_FB_DARK * float(np.percentile(V, 90)), 0.04, _FB_DARK))


def _run_len(cover: np.ndarray) -> int:
    """Leading entries of `cover` at or above _FB_COVER — the depth of the dark run."""
    bad = np.nonzero(cover < _FB_COVER)[0]
    return int(bad[0]) if bad.size else int(len(cover))


def frame_runs(V: np.ndarray):
    """(run depths in px per side, dark mask). Depths are 0 when the edge is not dark."""
    m = V < _dark_cut(V)
    col, row = m.mean(axis=0), m.mean(axis=1)
    return {"l": _run_len(col), "r": _run_len(col[::-1]),
            "t": _run_len(row), "b": _run_len(row[::-1])}, m


def feat_frame(rgb: np.ndarray):
    """19 features on dark runs anchored to the FRAME edge, not to the card."""
    f = dict.fromkeys(FB_NAMES, 0.0)
    h, w = rgb.shape[:2]
    V = rgb.mean(axis=2) / 255.0
    runs, m = frame_runs(V)

    f["fb_run_l"], f["fb_run_r"] = runs["l"] / w, runs["r"] / w
    f["fb_run_t"], f["fb_run_b"] = runs["t"] / h, runs["b"] / h
    norm = {"l": runs["l"] / w, "r": runs["r"] / w, "t": runs["t"] / h, "b": runs["b"] / h}
    f["fb_run_max"] = max(norm.values())
    f["fb_nsides"] = float(sum(v > 0.01 for v in norm.values()))
    f["fb_sym_lr"] = abs(norm["l"] - norm["r"])       # pillarbox is usually symmetric
    f["fb_sym_tb"] = abs(norm["t"] - norm["b"])

    side = max(norm, key=norm.get)
    d = runs[side]
    if d < 2:
        return [f[k] for k in FB_NAMES], list(FB_NAMES)

    if side == "l":
        sl, per_line = (slice(None), slice(0, d)), np.argmax(~m[:, :d + 1], axis=1)
    elif side == "r":
        sl, per_line = (slice(None), slice(w - d, w)), np.argmax(~m[:, w - d - 1:][:, ::-1], axis=1)
    elif side == "t":
        sl, per_line = (slice(0, d), slice(None)), np.argmax(~m[:d + 1, :], axis=0)
    else:
        sl, per_line = (slice(h - d, h), slice(None)), np.argmax(~m[h - d - 1:, :][::-1], axis=0)

    span = float(w if side in ("l", "r") else h)
    f["fb_extent"] = float(m[sl].mean())              # how solidly the run is filled
    f["fb_edge_jitter"] = float(per_line.std()) / span    # bar -> ~0, shadow -> ragged

    reg = _mask_of(sl, (h, w))
    core = _core(reg)                                 # interior only — see _core
    vd = V[core]
    f["fb_v"] = float(vd.mean())
    # How much darker the run is than everything else. fb_v alone is an EXPOSURE
    # statistic -- a dim photo makes every region dark -- which is why the loose
    # threshold could turn it into fake signal (see _dark_cut). The difference is
    # level-independent: a panel is far below its surroundings whatever the exposure,
    # a merely dim background is not.
    if (~reg).any():
        f["fb_contrast"] = float(np.median(V[~reg]) - vd.mean())
    ys, xs = np.nonzero(core)
    A = np.stack([xs, ys, np.ones(xs.size)], axis=1).astype(np.float64)
    sol, *_ = np.linalg.lstsq(A, vd.astype(np.float64), rcond=None)
    f["fb_flat"] = float((vd - A @ sol).std())

    res = ndimage.convolve(V, _KV, mode="reflect")[core]
    f["fb_detail"] = float(res.std())
    f["fb_kurt"] = float(kurtosis(res, fisher=True, bias=False)) if res.std() > _EPS else 0.0
    f["fb_struct"] = _struct_ratio(_local_std(V)[core])

    md = rgb[core].reshape(-1, 3).mean(axis=0)
    f["fb_chroma"] = float((md.max() - md.min()) / (md.max() + _EPS))
    f["fb_bluecast"] = float((md[2] - md[0]) / (md.mean() + _EPS))
    if (~reg).any():        # a uniformly dark frame leaves no "rest" to compare against
        rest = rgb[~reg].reshape(-1, 3).mean(axis=0)
        f["fb_rgbcos"] = float(md @ rest / (np.linalg.norm(md) * np.linalg.norm(rest) + _EPS))

    # step at the inner edge of the run: a panel boundary is a hard step
    gy, gx = np.gradient(V)
    if side in ("l", "r"):
        x = d if side == "l" else w - d - 1
        f["fb_inner_grad"] = float(np.abs(gx[:, max(0, min(w - 1, x))]).mean())
    else:
        yv = d if side == "t" else h - d - 1
        f["fb_inner_grad"] = float(np.abs(gy[max(0, min(h - 1, yv)), :]).mean())

    return [f[k] for k in FB_NAMES], list(FB_NAMES)


def _mask_of(sl, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[sl] = True
    return m


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
MACRO_NAMES = DK_NAMES + OC_NAMES + CD_NAMES + SR_NAMES + FB_NAMES


def feat_macro(rgb_full: np.ndarray):
    """All macro features. Input MUST be the full frame (use load_macro), not a crop.

    dk_* is kept alongside cd_*/sr_* on purpose: the next diag_macro run then gives a
    direct A/B between the blob-based group (known to lock onto the room) and the
    card-anchored one, instead of us guessing which to keep.
    """
    if min(rgb_full.shape[:2]) < 8:
        # No macro geometry is measurable on a frame this small, and np.gradient needs
        # at least 2 samples per axis. load_macro never produces one from a real photo;
        # this only guards a corrupt or 1-pixel input.
        return [0.0] * len(MACRO_NAMES), list(MACRO_NAMES)
    fd, nd = feat_dark(rgb_full)
    fo, no = feat_overcrop(rgb_full)
    fc, nc = feat_card(rgb_full)
    fb, nb = feat_frame(rgb_full)
    return fd + fo + fc + fb, nd + no + nc + nb
