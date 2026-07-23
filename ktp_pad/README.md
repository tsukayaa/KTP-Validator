# Stage 3 — KTP screen-replay detector (forensics + LightGBM)

KTP **asli** vs **spoof** (photo of a KTP shown on a screen → moiré / glare "pendar" / bezel).
600/600 labeled, CPU, **no pretrained weights**, fully offline.

## Why not CLIP
Screen-replay is a **low-level texture** signal. CLIP resizes to 224×224 and is trained to be
**invariant** to resize / JPEG / color shift — exactly the recapture cues we need — so a CLIP
fine-tune is structurally blind to it (that is why it stalled at F1 ≈ 0.74). We instead compute
the texture forensics CLIP discards and feed a small gradient-boosted tree.

Features (212 total, two views of each file):

| view | how it is read | groups |
| --- | --- | --- |
| **texture** (125) | native-resolution **center crop** — resizing destroys the signal | FFT moiré peaks (strongest), SRM high-pass residuals, specular/glare blobs, uniform LBP micro-texture, colour banding, JPEG blockiness |
| **macro** (87) | **full frame** downscaled to 512 — a center crop would cut off the border evidence | `dk_*` dark blob (26), `oc_*` frame ring (14), `cd_*` card box (12), `sr_*` card-anchored surround (16), `fb_*` frame-edge runs (19) — all in `features_macro.py` |

The four macro localisers answer different questions and **disagree on real images** —
that is the point of keeping all of them until `diag_macro.py` says which earn their place.

### Round 1 result: `dk_*` measured the wrong thing

The premise was that both classes have a dark region — spoof = the monitor that was never
cropped out, genuine = the **shadow cast by the photographer** — so `dk_*` measured the
physics that separates a shadow from a display surface rather than "is it dark".

The AUCs looked healthy. **The mask overlays showed they were measuring the room.**
`largest_border_blob` locks onto the wall / floor / hoodie / bed on genuine images, and a
mix of monitor, hand, shadow and desk on spoof. So those numbers are room-background
statistics, the physics hypothesis is **untested rather than refuted**, and the strongest of
them (`dk_v_p50` = "spoof surroundings are darker") is a plausible **acquisition-environment
shortcut** — screen-photographers sit at a desk in a dim room, genuine photos get taken on a
bed or a floor. That collapses across users and in production.

Take the lesson, not just the fix: **a good AUC from a mislocalised region is worse than a
bad one**, because it reads as progress.

### Round 2: `cd_*` / `sr_*`, anchored to the card

The data pointed at the fix — `oc_ring_chi2` (0.740) and `oc_ring_v_ratio` (0.245 ≡ 0.755)
were among the strongest features and use **no blob detection at all**, just an annulus.
So: locate the card, then measure the strips immediately around it, where an uncropped
screen actually sits.

`find_card` uses **print structure**, not colour (KTP blue varies with print run, fading,
white balance and photocopies; a displayed card takes the panel's gamut). It must be
class-**neutral** — a real card and a displayed one both carry print — and `diag_macro.py`
checks that by reporting `cd_found` rates per class.

`sr_dk_*` is computed on the **single darkest** of the four surround strips, not their
average: an uncropped monitor shows on one or two sides only, so averaging dilutes it away.
`sr_side_contrast` carries the asymmetry itself.

**`dk_*` is retained** so the next `diag_macro` run is a direct A/B rather than a guess.

### Round 3: `fb_*`, because `sr_*` is blind to the pillarbox case

A real example from the data: the KTP sits in the **middle of a bright scene**, and the
uncropped monitor appears only as thin black **pillarbox bars along the far left and right
of the frame**. None of the earlier groups sees that properly — `sr_*` hugs the card, so its
strips land on the bright background *between* the card and the bars and miss them entirely;
`oc_*` averages a ring over all four sides, so a bar on one side is diluted.

So `fb_*` scans inward from each **frame** edge for a contiguous mostly-dark run. The
discriminator against a photographer's shadow reaching the edge is not darkness but
**geometry**: a pillarbox bar spans the full height at constant width (`fb_edge_jitter` ≈ 0,
`fb_extent` ≈ 1), a shadow's boundary is ragged and varies. The same shadow-vs-panel physics
as `sr_*` is then applied to the run interior.

Measured on the synthetic pillarbox-vs-edge-shadow pair, `sr_dk_v` came out **0.7057 vs
0.7058 vs 0.7055** across pillarbox / edge-shadow / clean — identical to four decimals. That
is `sr_*` demonstrating it cannot see the case at all, and it is why `fb_*` exists.

### Both groups are the spec §4.C bezel shortcut

Before quoting any gain: ablate the block and run the bezel-crop diagnostic. A large gain
that vanishes when the image is cropped means the model learned the shortcut, not the
phenomenon — report that, don't hide it.

`oc_*`/`cd_*` over-crop ("KTP di-zoom sampai tidak ada background") is deliberately a **weak
feature, not a gate**. Honest users crop tightly too, so treating over-crop as evidence of
spoof false-rejects real customers. It is a mild prior only — a fraudster cropping the
monitor away produces an over-cropped frame, so it partly covers what `sr_*` then loses.

## Data (project root)
```
data2/positives/   KTP asli (genuine)         -> label 1
data2/negatives/   spoof (pendar/bezel)       -> label 0
```
Class convention: **positives = asli = 1**, negatives = spoof = 0.

## Run
```bash
pip install -r ktp_pad/requirements.txt      # numpy scipy lightgbm scikit-learn pillow
python ktp_pad/split_dataset.py              # -> data2/train, data2/test (100/class test)
python ktp_pad/diag_macro.py                 # RUN FIRST — validates the dk_*/oc_* features
python ktp_pad/train_pad.py                  # train + eval on test
```

`diag_macro.py` prints the `dk_present`, `cd_found` and `fb_run_max>0` contingencies,
per-feature AUC **within each gate's own subpopulation** (the `all` column is inflated by the
gate itself — don't quote it), and writes four-panel overlays to `output/macro_masks/`:
original | `dk_*` blob | `cd_*` card (green) + surround strips (blue, **darkest in red** —
the strip every `sr_dk_*` value comes from) | `fb_*` frame-edge runs (orange).
`*_nocard.png` are localisation failures.

**Look at the pictures before reading the table.** That is what caught round 1.
Watch `cd_found` specifically: it is supposed to be class-neutral, so if its two rates
differ by more than a few points the localiser is itself class-dependent and every `sr_*`
number inherits that bias.

## Outputs
```
models/pad_lgbm.txt        trained LightGBM booster
models/pad_meta.json       threshold + feature names + class map
output/results.json        per-file scores + confusion (TP/FP/TN/FN) + metrics
output/positives/          test images PREDICTED asli
output/negatives/          test images PREDICTED spoof
output/macro_diag.md       diag_macro.py: contingency + per-feature AUC
output/macro_masks/        diag_macro.py: dark-region overlays, for eyeballing
```
`train_pad.py` prints accuracy / precision / recall / F1(pos) / F1(neg) / **F1-macro** and the
delta vs the CLIP baseline (0.74). Threshold is tuned on a val slice carved from train — the
test set is never used for tuning.

## Honest caveats
- **No `user_id` grouping.** If the same physical card appears in both train and test, the
  number is optimistic. If you get per-submission user ids, split by user, not by file.
- 600 spoofs are "caught" spoofs → the model measures re-catching known-style spoofs, not
  unseen sophisticated ones. Treat the F1 as an upper bound.
- If `train_pad.py` warns median resolution < 1 MP, the texture signal is compression-damaged;
  the fix is the capture pipeline, not the model.
- **`dk_*` is known to mislocalise on the real data** (round 1, above). Its AUCs describe the
  room, not the monitor or the shadow. Do not quote them.
- **`cd_*`/`sr_*`/`fb_*` have only been checked against synthetic scenes** — a card composited
  onto six backgrounds (dark panel, smooth desk, textured surface, carpet, bright wall, card
  near the frame edge), plus a pillarbox-vs-edge-shadow pair, plus nine degenerate inputs
  (all-black, all-white, 1-pixel, pure noise) for NaN/crash safety. Localisation lands within
  6 px of the true card edge on all six backgrounds and the pillarbox pair separates 10/10 on
  direction. That confirms the code works, **not** that the features separate the classes.
  Nothing here should be quoted until `diag_macro.py` has run on the real data and the
  overlays have been checked.
- **`find_card` known limitation:** when the card fills the entire frame there is no
  card-vs-background split for Otsu to find, so it splits print-vs-margin and the box comes
  back ~40 px short on each side. The features still move the right way, but `cd_margin_*` is
  not a calibrated distance.
