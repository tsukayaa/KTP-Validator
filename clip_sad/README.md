# CLIP Deep-SAD — Stage-2 KTP-only reject gate

Reference implementation for the office-side pipeline. Stage-1 (zero-shot CLIP,
"is a KTP present?") is **elsewhere**; this folder is only **Stage-2**: given an
image that already contains a KTP, decide

```
KTP-only (front, no person)                       -> "yes"
selfie+KTP / person+KTP / KTP back / SIM / ATM /  -> "no"
other cards / unknown reject cases
```

## Stage 1 prompts (zero-shot "is a KTP present?")

`stage1_prompts.py` holds the **pre-selfie-attack** prompt pools (baru.md
Appendix B). `yes` = a KTP is present (selfie+KTP included on purpose -> it
passes Stage 1 and is rejected by Stage 2). `no` = no card + paper-document
hard-negatives (KK/akte). Aggregation is **max per pool**; text encoded once.

```bash
python -m clip_sad.stage1_prompts some_image.jpg   # KTP present? -> pass/reject
```

## Why Deep SAD (not a binary classifier)

- Positives: ~5000, tight & well-sampled. Negatives: ~500, **open-ended** (many
  real reject cases are not in the set).
- A binary classifier overfits the 500 seen negatives and is undefined on unseen
  ones.
- Deep SAD models the **positive manifold**: pull positives to a center `c`,
  push the few labeled negatives away. At inference, **distance to `c`** is the
  score — anything far is rejected, including negative types never trained on.
- `selfie+KTP` is the hard case (KTP dominates the embedding, sits near the
  positive manifold). It gets a heavier loss weight (`cfg.selfie_weight`).

## Two versions

| | file | encoder | time (6 CPU) | selfie+KTP |
|---|---|---|---|---|
| **A** fast baseline | `train_a.py` | **frozen**, features cached once | ~15 min | weak |
| **B** full fine-tune | `train_b.py` | last `N` ViT blocks trained | ~9 h (≈1–2 h @ ~30 vCPU) | best shot |

Inference (`predict.py`) is **identical** for both: `embed -> distance to c -> threshold`.

## Data layout (edit names/root in `sad_common.py: SADConfig`)

```
data/train/positive/         KTP-only images               (label +1)
data/train/negative/         mixed negatives               (label -1)
data/train/negative_selfie/  selfie+KTP, heavier weight    (label -1)  [optional]
data/test/positive/          held-out KTP-only             (label +1)
data/test/negative/          held-out mixed negatives      (label -1)
data/test/negative_selfie/   held-out selfie+KTP           (label -1)  [optional]
weights/ViT-B-16.pt          OFFLINE OpenAI CLIP checkpoint
```

The model is trained and the threshold is **calibrated on `train/`**; `test/` is
a **held-out** set used only for the printed `[test]` accept/reject numbers, so
those numbers reflect generalization, not memorization.

**Minimal labeling** (≈1 h): you do **not** need to split negatives by every
type. Only pull the `selfie+KTP` images into `negative_selfie/`; leave SIM/ATM/
back/etc. mixed in `negative/`. If `negative_selfie/` is empty, version B
degrades gracefully to plain SAD (no extra weighting). `test/` may be small (a
few hundred each) — it only scores, it doesn't train.

## Run

```bash
pip install git+https://github.com/openai/CLIP.git    # + torch, pillow, numpy
python -m clip_sad.train_a                              # fast
python -m clip_sad.train_b                              # full
python -m clip_sad.predict some_image.jpg --model models/clip_sad_B.pt
python -m clip_sad.predict some_folder/   --model models/clip_sad_A.pt
```

## Caveats (honest)

- **Collapse**: Deep SVDD/SAD can cheat by mapping everything to `c`. Mitigated
  with bias-free head layers + weight decay + fixed center from init. If train
  loss → ~0 and *everything* scores "yes", that's collapse — raise weight decay
  / lower LR.
- **Residual hole**: a selfie where the KTP fills the frame and the face is a
  tiny sliver may still pass. Whole-image embeddings can't fully close this;
  the robust closer is a separate face-size gate (out of scope here).
- CPU `.pt` is fp32 after `clip.load(..., device="cpu")` — fine for training.
- Bump GKE vCPUs to cut version-B epoch time roughly linearly.
```
