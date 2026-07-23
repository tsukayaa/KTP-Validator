# Stage 3 — KTP screen-replay detector (forensics + LightGBM)

KTP **asli** vs **spoof** (photo of a KTP shown on a screen → moiré / glare "pendar" / bezel).
600/600 labeled, CPU, **no pretrained weights**, fully offline.

## Why not CLIP
Screen-replay is a **low-level texture** signal. CLIP resizes to 224×224 and is trained to be
**invariant** to resize / JPEG / color shift — exactly the recapture cues we need — so a CLIP
fine-tune is structurally blind to it (that is why it stalled at F1 ≈ 0.74). We instead compute
the texture forensics CLIP discards and feed a small gradient-boosted tree.

Features (165 total, two views of each file):

| view | how it is read | groups |
| --- | --- | --- |
| **texture** (125) | native-resolution **center crop** — resizing destroys the signal | FFT moiré peaks (strongest), SRM high-pass residuals, specular/glare blobs, uniform LBP micro-texture, colour banding, JPEG blockiness |
| **macro** (40) | **full frame** downscaled to 512 — a center crop would cut off the border evidence | `dk_*` dark-region analysis, `oc_*` over-crop proxies (`features_macro.py`) |

### The `dk_*` group is NOT "is there a black area"

Both classes have a dark region: spoof = the monitor that was never cropped out,
genuine = the **shadow cast by the photographer**. A "dark blob ⇒ spoof" feature is worse
than useless — it costs BPCER. So `dk_*` measures what physically separates a cast shadow
from a display surface: `dk_struct` (a shadow still contains the surface texture, a dark
panel contains only sensor noise), `dk_rgb_cos` (same material ⇒ mean RGB stays collinear
with the lit region), `dk_edge_rise` / `dk_penumbra` (shadows have a penumbra, screen edges
are a step), `dk_edge_inlier` (screen boundary is a straight line, a shadow's curves) and
`dk_inner_frac` (a shadow can fall across the card; screen area around a displayed card
cannot). All level-independent, since black level varies wildly across panels.

**This is the bezel shortcut of spec §4.C.** Before quoting any gain from it: run the
ablation with the `dk_*` block dropped, and run the bezel-crop diagnostic. A large gain that
vanishes when the image is cropped means the model learned the shortcut, not the phenomenon —
report that, don't hide it.

`oc_*` (over-crop / "KTP di-zoom sampai tidak ada background") is deliberately a **weak
feature, not a gate**. Honest users crop tightly too, so treating over-crop as evidence of
spoof false-rejects real customers. It is there as a mild prior only — a fraudster cropping
the monitor away produces an over-cropped frame, so it partly covers what `dk_*` then loses.

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

`diag_macro.py` answers the only question that matters for the `dk_*` group: **among images
that have a dark region, do the features separate shadow from screen?** It prints the
`dk_present` × label contingency, per-feature AUC *within that subpopulation* (the full-set
AUC is inflated by `dk_present` itself — don't quote it), and writes mask overlays to
`output/macro_masks/` so you can check the detector found the monitor/shadow and not the
KTP's own photo box. If one side of the contingency is tiny, the tree will just learn
"dark ⇒ spoof" and false-reject every shadowed genuine — that is a data-collection problem,
not a tuning problem.

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
- **The `dk_*`/`oc_*` features have only been checked against synthetic images** (a flat
  neutral strip vs a multiplicative soft-edged falloff) — enough to confirm the code runs and
  each feature points the intended way, **not** enough to claim it works. `dk_detail_ratio_gain`
  already failed that check and is kept only as a candidate. Nothing about this group should be
  quoted until `diag_macro.py` has been run on the real data.
