"""
CODE 1 — split dataset.

Reads  data2/positives  (KTP asli / genuine)  and  data2/negatives (spoof: pendar/bezel),
copies them (non-destructive) into:

    data2/train/positives   data2/train/negatives
    data2/test/positives    data2/test/negatives

Default: N_TEST per class -> test, the rest -> train.  Fixed seed = reproducible.

Run from the project root (or anywhere):
    python ktp_pad/split_dataset.py
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # project root
DATA = ROOT / "data2"
CLASSES = ("positives", "negatives")               # positives = asli, negatives = spoof
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

N_TEST = 100          # images per class held out for test
SEED = 42


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in EXTS)


def main():
    random.seed(SEED)

    for cls in CLASSES:
        src = DATA / cls
        if not src.exists():
            sys.exit(f"missing source folder: {src}")

    # fresh train/test dirs
    for split in ("train", "test"):
        for cls in CLASSES:
            (DATA / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        imgs = list_images(DATA / cls)
        random.shuffle(imgs)
        n = len(imgs)
        if n < N_TEST + 10:
            print(f"WARN {cls}: only {n} images (need > {N_TEST})")
        test = imgs[:N_TEST]
        train = imgs[N_TEST:]
        for p in test:
            shutil.copy2(p, DATA / "test" / cls / p.name)
        for p in train:
            shutil.copy2(p, DATA / "train" / cls / p.name)
        print(f"{cls:10s}: {n} total  ->  train {len(train)} / test {len(test)}")

    print(f"done. splits under {DATA}/train and {DATA}/test  (seed={SEED})")


if __name__ == "__main__":
    main()
