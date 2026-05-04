"""CLI: Train and evaluate the XGBoost approach with handcrafted CV features.

Usage:
    python run_xgboost.py
    python run_xgboost.py --data_dir ./data --n_estimators 300
"""
import argparse
import logging
import sys
from pathlib import Path

from config import DataConfig, XGBoostConfig
from shared.data_loader import DatasetLoader, InsufficientDataError
from shared.metrics import MetricsCalculator
from shared.splitter import DataSplitter
from shared.visualizer import ResultVisualizer
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
        description="Train XGBoost binary card classifier with CV features"
    )
    parser.add_argument("--data_dir", default="./data", help="Root data directory")
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--results_dir", default="./results/xgboost", help="Where to save plots")
    parser.add_argument("--no_cache", action="store_true", help="Ignore feature cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(data_dir=args.data_dir)
    model_cfg = XGBoostConfig()

    if args.n_estimators is not None:
        model_cfg.n_estimators = args.n_estimators
    if args.max_depth is not None:
        model_cfg.max_depth = args.max_depth
    if args.learning_rate is not None:
        model_cfg.learning_rate = args.learning_rate

    logger.info("=== XGBoost Card Classifier ===")
    logger.info("Data dir       : %s", data_cfg.data_dir)
    logger.info("n_estimators   : %d", model_cfg.n_estimators)
    logger.info("max_depth      : %d", model_cfg.max_depth)
    logger.info("learning_rate  : %s", model_cfg.learning_rate)

    try:
        loader = DatasetLoader(data_cfg)
        paths, labels = loader.load()
    except InsufficientDataError as exc:
        logger.error("Dataset error: %s", exc)
        sys.exit(1)

    if args.no_cache:
        import shutil
        cache_dir = Path(model_cfg.feature_cache_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("Feature cache cleared")

    splitter = DataSplitter(data_cfg)
    split_result = splitter.split(paths, labels)

    extractor = ImageFeatureExtractor(model_cfg)
    trainer = XGBTrainer(model_cfg)
    results = trainer.train(split_result, extractor)

    calc = MetricsCalculator()
    calc.print_report(results, "XGBoost + CV Features")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    viz = ResultVisualizer()

    cm_key = "confusion_matrix_sum" if split_result.strategy == "cv" else "confusion_matrix"
    if cm_key in results:
        viz.plot_confusion_matrix(
            results[cm_key],
            "XGBoost — Confusion Matrix",
            str(results_dir / "confusion_matrix.png"),
        )

    if "feature_importance" in results and results["feature_importance"]:
        viz.plot_feature_importance(
            results["feature_importance"],
            str(results_dir / "feature_importance.png"),
        )

    logger.info("Done. Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
