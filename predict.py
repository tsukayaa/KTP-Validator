"""CLI: Single-image prediction with either or both approaches.

Usage:
    python predict.py <image_path> --approach mobilenet
    python predict.py <image_path> --approach xgboost
    python predict.py <image_path> --approach mobilenet --tta
    python predict.py <image_path> --approach xgboost --tta
    python predict.py <image_path> --approach both
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from config import MobileNetConfig, XGBoostConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_BORDER = "═" * 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict card/no-card for a single image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument(
        "--approach",
        choices=["mobilenet", "xgboost", "both"],
        default="mobilenet",
        help="Which model to use",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable Test-Time Augmentation (ignored when --approach=both; always applied)",
    )
    parser.add_argument(
        "--mobilenet_model",
        default="./models/mobilenet/best_model.pth",
        help="Path to saved MobileNet weights",
    )
    parser.add_argument(
        "--xgboost_model",
        default="./models/xgboost/model.json",
        help="Path to saved XGBoost model",
    )
    return parser.parse_args()


def _print_result(result: dict, approach_label: str, elapsed_ms: float) -> None:
    print(_BORDER)
    print(f"  Prediction: {result['class']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Approach  : {approach_label}")
    if result.get("tta_applied") or result.get("rotation_augment"):
        print("  TTA       : enabled")
    print(f"  Time      : {elapsed_ms:.0f}ms")
    print(_BORDER)


def run_mobilenet(image_path: str, tta: bool, model_path: str) -> None:
    from approach1_mobilenet.predictor import MobileNetPredictor

    cfg = MobileNetConfig()
    predictor = MobileNetPredictor(model_path, cfg)

    t0 = time.perf_counter()
    result = predictor.predict_with_tta(image_path) if tta else predictor.predict(image_path)
    elapsed = (time.perf_counter() - t0) * 1000

    label = "MobileNetV3 (TTA)" if tta else "MobileNetV3"
    _print_result(result, label, elapsed)


def run_xgboost(image_path: str, tta: bool, model_path: str) -> None:
    from approach2_xgboost.predictor import XGBPredictor

    cfg = XGBoostConfig()
    predictor = XGBPredictor(model_path, cfg)

    t0 = time.perf_counter()
    result = (
        predictor.predict_with_rotation_augment(image_path)
        if tta
        else predictor.predict(image_path)
    )
    elapsed = (time.perf_counter() - t0) * 1000

    label = "XGBoost (rotation TTA)" if tta else "XGBoost"
    _print_result(result, label, elapsed)


def run_both(image_path: str, mobilenet_model: str, xgboost_model: str) -> None:
    from approach1_mobilenet.predictor import MobileNetPredictor
    from approach2_xgboost.predictor import XGBPredictor

    mn_cfg = MobileNetConfig()
    xgb_cfg = XGBoostConfig()

    mn_pred = MobileNetPredictor(mobilenet_model, mn_cfg)
    xgb_pred = XGBPredictor(xgboost_model, xgb_cfg)

    t0 = time.perf_counter()
    mn_result = mn_pred.predict_with_tta(image_path)
    mn_elapsed = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    xgb_result = xgb_pred.predict_with_rotation_augment(image_path)
    xgb_elapsed = (time.perf_counter() - t0) * 1000

    _print_result(mn_result, "MobileNetV3 (TTA)", mn_elapsed)
    _print_result(xgb_result, "XGBoost (rotation TTA)", xgb_elapsed)

    if mn_result["label"] == xgb_result["label"]:
        print(f"  ✓ Both models agree: {mn_result['class']}")
    else:
        print(
            f"  ⚠ Models disagree — MobileNet: {mn_result['class']}  "
            f"XGBoost: {xgb_result['class']}"
        )


def main() -> None:
    args = parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: image not found at '{args.image_path}'")
        sys.exit(1)

    try:
        if args.approach == "mobilenet":
            run_mobilenet(args.image_path, args.tta, args.mobilenet_model)
        elif args.approach == "xgboost":
            run_xgboost(args.image_path, args.tta, args.xgboost_model)
        else:
            run_both(args.image_path, args.mobilenet_model, args.xgboost_model)
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        logger.exception("Prediction error")
        sys.exit(1)


if __name__ == "__main__":
    main()
