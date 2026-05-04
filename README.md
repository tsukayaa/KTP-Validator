# KTP Selfie Binary Classifier

Layer 1 of a 2-layer identity-card pipeline: **is there an identity card in this selfie?**

Two independent approaches, both CPU-only, inference < 10s:

| Approach | Method | Key strength |
|---|---|---|
| 1 | MobileNetV3-Small (transfer learning) | High accuracy, handles subtle visual cues |
| 2 | XGBoost + handcrafted CV features | Fast, interpretable, no GPU needed |

---

## Dataset layout

```
data/
├── positives/    # selfies WITH an identity card visible
└── negatives/    # selfies WITHOUT any card, random photos, etc.
```

Supported formats: `.jpg` `.jpeg` `.png` `.bmp` `.webp`

---

## Quick start

```bash
pip install -r requirements.txt

# Train MobileNet
python run_mobilenet.py --data_dir ./data

# Train XGBoost
python run_xgboost.py --data_dir ./data

# Compare both (trains both, prints side-by-side table)
python compare.py --data_dir ./data

# Predict on a single image
python predict.py photo.jpg --approach mobilenet --tta
python predict.py photo.jpg --approach xgboost --tta
python predict.py photo.jpg --approach both
```

---

## Splitting strategy

| Dataset size | Strategy |
|---|---|
| < 200 images | 5-Fold Stratified Cross-Validation |
| ≥ 200 images | Fixed 70 / 20 / 10 train / val / test split |

Logged at runtime with reasoning.

---

## Approach 1 — MobileNetV3-Small

- Pretrained on ImageNet (weights downloaded on first run, ~10 MB, one-time only)
- Backbone features frozen by default; only the 2-layer classifier head trains
- Heavy augmentation: full 360° rotation, grayscale, jitter, affine, perspective, blur
- Small-dataset oversampling via `repeat_factor` (default 10× per epoch)
- Class-weighted CrossEntropyLoss for imbalanced datasets
- Early stopping on validation loss (patience 7 epochs)
- **TTA at inference**: averages predictions at 0°/90°/180°/270° for robustness

**CLI options:**

```
python run_mobilenet.py [--data_dir PATH] [--epochs N] [--batch_size N]
                        [--learning_rate F] [--no_freeze] [--results_dir PATH]
```

---

## Approach 2 — XGBoost + Handcrafted Features

All features are **rotation-invariant** and **colour-agnostic** (computed on grayscale):

| Feature group | Description | Approx. size |
|---|---|---|
| LBP texture | Uniform LBP at radii 1/2/3 | ~54 |
| Edge/line | Canny density, Sobel magnitude, HoughLinesP count | 5 |
| DCT frequency | Low/mid/high energy ratios, spectral entropy | 5 |
| GLCM texture | Contrast, dissimilarity, homogeneity, energy, correlation | 15 |
| Shape | Rectangle contour count, area ratio, aspect ratio, solidity | 5 |
| Statistics | Mean, std, skewness, kurtosis, entropy, intensity range | 6 |

Feature extraction results are **cached** to `./cache/` — subsequent runs skip extraction if the image set is unchanged.

**CLI options:**

```
python run_xgboost.py [--data_dir PATH] [--n_estimators N] [--max_depth N]
                      [--learning_rate F] [--results_dir PATH] [--no_cache]
```

---

## Single-image prediction

```
python predict.py <image_path> --approach {mobilenet,xgboost,both} [--tta]
                  [--mobilenet_model PATH] [--xgboost_model PATH]
```

Output example:
```
════════════════════════════════════════════
  Prediction: HAS_CARD
  Confidence: 0.9231
  Approach  : MobileNetV3 (TTA)
  TTA       : enabled
  Time      : 412ms
════════════════════════════════════════════
```

---

## Output artefacts

```
models/
├── mobilenet/best_model.pth
└── xgboost/model.json

results/
├── mobilenet/
│   ├── confusion_matrix.png
│   └── training_history.png
├── xgboost/
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── comparison.png

cache/
├── features.npy
├── labels.npy
└── cache_key.txt
```

---

## Design notes

- **CPU-only**: `torch.device("cpu")` is hardcoded — no CUDA calls anywhere.
- **B&W images**: MobileNet path calls `image.convert("RGB")` (grayscale → 3-channel). XGBoost path operates on grayscale anyway.
- **Rotation robustness**: MobileNet — 360° training augmentation + TTA. XGBoost — LBP uniform method is rotation-invariant + rotation-augmented inference.
- **Class imbalance**: auto-detected; class-weighted loss (MobileNet) and `scale_pos_weight` (XGBoost) applied automatically.
- **Reproducibility**: `random_state=42` everywhere; all hyperparameters logged at training start.
- **CLIP baseline**: Zero-shot CLIP achieves ~86% accuracy on this task. Use that as a reference threshold.
