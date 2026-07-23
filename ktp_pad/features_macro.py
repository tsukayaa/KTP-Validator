"""
CODE 2b — MACRO-cue features: dark-region discrimination + over-crop proxies.

WHY THIS EXISTS, AND WHY IT IS A SEPARATE MODULE FROM train_pad.py:

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
        loc = np.sqrt(np.clip(ndimage.uniform_filter(V ** 2, 5) -
                              ndimage.uniform_filter(V, 5) ** 2, 0, None))
        ld = loc[ero]
        f["dk_localstd"] = float(ld.mean())
        # Structure vs white noise — LEVEL-INDEPENDENT, so it survives whatever the
        # black level happens to be. Real texture is spatially clustered (flat areas
        # plus sharp edges) -> tall p95/p50 and heavy-tailed residual. Sensor noise is
        # uniform -> p95/p50 ~1.3, kurtosis ~0.
        f["dk_struct"] = float(np.percentile(ld, 95)) / (float(np.percentile(ld, 50)) + _EPS)
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
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
MACRO_NAMES = DK_NAMES + OC_NAMES


def feat_macro(rgb_full: np.ndarray):
    """All macro features. Input MUST be the full frame (use load_macro), not a crop."""
    fd, nd = feat_dark(rgb_full)
    fo, no = feat_overcrop(rgb_full)
    return fd + fo, nd + no
