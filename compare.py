"""CLI: Train both approaches on identical splits and compare side-by-side.

Usage:
    python compare.py --data_dir ./data
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

from config import DataConfig, MobileNetConfig, XGBoostConfig
from shared.data_loader import DatasetLoader, InsufficientDataError
from shared.metrics import MetricsCalculator
from shared.splitter import DataSplitter
from shared.visualizer import ResultVisualizer
from approach1_mobilenet.trainer import MobileNetTrainer
from approach2_xgboost.feature_extractor import ImageFeatureExtractor
from approach2_xgboost.trainer import XGBTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train both classifiers on the same splits and compare results"
    )
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument(
        "--skip_mobilenet", action="store_true", help="Skip MobileNet (faster, for quick XGB eval)"
    )
    return parser.parse_args()


def _fmt_metric(results: dict, key: str) -> str:
    val = results.get(key)
    if val is None:
        return "  N/A  "
    if isinstance(val, dict):
        return f"{val['mean']:.4f} ± {val['std']:.4f}"
    return f"{float(val):.4f}       "


def _model_size_mb(path: str) -> str:
    if not Path(path).exists():
        return "N/A"
    return f"{Path(path).stat().st_size / (1024 * 1024):.2f} MB"


def _measure_inference(approach: str, model_cfg, image_path: str, n_runs: int = 5) -> float:
    """Return average inference time in ms over n_runs."""
    times = []
    if approach == "mobilenet":
        from approach1_mobilenet.predictor import MobileNetPredictor
        model_path = str(Path(model_cfg.model_save_dir) / "best_model.pth")
        if not Path(model_path).exists():
            return float("nan")
        pred = MobileNetPredictor(model_path, model_cfg)
        fn = pred.predict
    else:
        from approach2_xgboost.predictor import XGBPredictor
        if not Path(model_cfg.model_save_path).exists():
            return float("nan")
        pred = XGBPredictor(model_cfg.model_save_path, model_cfg)
        fn = pred.predict

    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(image_path)
        times.append((time.perf_counter() - t0) * 1000)

    return float(sum(times) / len(times))


def print_comparison(mn_results: dict, xgb_results: dict) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    print("\n" + "═" * 58)
    print("  Approach Comparison")
    print("═" * 58)
    print(f"  {'Metric':<14}  {'MobileNetV3':<22}  {'XGBoost':<22}")
    print("─" * 58)
    for key, label in zip(metrics, metric_labels):
        mn_str = _fmt_metric(mn_results, key)
        xgb_str = _fmt_metric(xgb_results, key)
        print(f"  {label:<14}  {mn_str:<22}  {xgb_str:<22}")
    print("═" * 58)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(data_dir=args.data_dir)
    mn_cfg = MobileNetConfig()
    xgb_cfg = XGBoostConfig()

    try:
        loader = DatasetLoader(data_cfg)
        paths, labels = loader.load()
    except InsufficientDataError as exc:
        logger.error("Dataset error: %s", exc)
        sys.exit(1)

    splitter = DataSplitter(data_cfg)
    split_result = splitter.split(paths, labels)

    all_results: dict = {}

    # ── MobileNet ────────────────────────────────────────────────────
    if not args.skip_mobilenet:
        logger.info("Training MobileNetV3-Small …")
        mn_trainer = MobileNetTrainer(mn_cfg)
        mn_results = mn_trainer.train(split_result)
        all_results["MobileNetV3"] = mn_results
        MetricsCalculator().print_report(mn_results, "MobileNetV3-Small")
    else:
        mn_results = {}
        logger.info("MobileNet skipped (--skip_mobilenet)")

    # ── XGBoost ──────────────────────────────────────────────────────
    logger.info("Training XGBoost …")
    extractor = ImageFeatureExtractor(xgb_cfg)
    xgb_trainer = XGBTrainer(xgb_cfg)
    xgb_results = xgb_trainer.train(split_result, extractor)
    all_results["XGBoost"] = xgb_results
    MetricsCalculator().print_report(xgb_results, "XGBoost + CV Features")

    # ── Side-by-side printout ─────────────────────────────────────────
    print_comparison(mn_results, xgb_results)

    # ── Inference timing (fixed-split only — need a sample image) ────
    if split_result.strategy == "fixed" and split_result.test:
        sample_image = split_result.test[0][0]
        if not args.skip_mobilenet:
            mn_ms = _measure_inference("mobilenet", mn_cfg, sample_image)
            xgb_ms = _measure_inference("xgboost", xgb_cfg, sample_image)
            mn_size = _model_size_mb(str(Path(mn_cfg.model_save_dir) / "best_model.pth"))
            xgb_size = _model_size_mb(xgb_cfg.model_save_path)
            print(f"  {'Inference':<14}  {f'{mn_ms:.0f}ms':<22}  {f'{xgb_ms:.0f}ms':<22}")
            print(f"  {'Model Size':<14}  {mn_size:<22}  {xgb_size:<22}")
            print("═" * 58)

    # ── Comparison plot ───────────────────────────────────────────────
    if all_results:
        viz = ResultVisualizer()
        viz.plot_comparison(
            all_results,
            str(results_dir / "comparison.png"),
        )

    logger.info("Comparison complete. Plots saved to %s", results_dir)


if __name__ == "__main__":
    main()
