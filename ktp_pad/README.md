# Stage 3 — KTP screen-replay detector (forensics + LightGBM)

KTP **asli** vs **spoof** (photo of a KTP shown on a screen → moiré / glare "pendar" / bezel).
600/600 labeled, CPU, **no pretrained weights**, fully offline.

## Why not CLIP
Screen-replay is a **low-level texture** signal. CLIP resizes to 224×224 and is trained to be
**invariant** to resize / JPEG / color shift — exactly the recapture cues we need — so a CLIP
fine-tune is structurally blind to it (that is why it stalled at F1 ≈ 0.74). We instead compute
the texture forensics CLIP discards and feed a small gradient-boosted tree.

Features: FFT moiré peaks (strongest), SRM high-pass residuals, specular/glare blobs, uniform
LBP micro-texture, colour-banding, JPEG double-compression blockiness (~120 features).

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
python ktp_pad/train_pad.py                  # train + eval on test
```

## Outputs
```
models/pad_lgbm.txt        trained LightGBM booster
models/pad_meta.json       threshold + feature names + class map
output/results.json        per-file scores + confusion (TP/FP/TN/FN) + metrics
output/positives/          test images PREDICTED asli
output/negatives/          test images PREDICTED spoof
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
