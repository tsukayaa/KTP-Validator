"""CLI: Train and evaluate the MobileNetV3-Small approach.

Usage:
    python run_mobilenet.py
    python run_mobilenet.py --data_dir ./data --epochs 50 --batch_size 16
"""
import argparse
import logging
import sys
from pathlib import Path

from config import DataConfig, MobileNetConfig
from shared.data_loader import DatasetLoader, InsufficientDataError
from shared.metrics import MetricsCalculator
from shared.splitter import DataSplitter
from shared.visualizer import ResultVisualizer
from approach1_mobilenet.trainer import MobileNetTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3-Small binary card classifier"
    )
    parser.add_argument("--data_dir", default="./data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override LR")
    parser.add_argument("--no_freeze", action="store_true", help="Unfreeze backbone")
    parser.add_argument("--results_dir", default="./results/mobilenet", help="Where to save plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(data_dir=args.data_dir)
    model_cfg = MobileNetConfig()

    if args.epochs is not None:
        model_cfg.epochs = args.epochs
    if args.batch_size is not None:
        model_cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        model_cfg.learning_rate = args.learning_rate
    if args.no_freeze:
        model_cfg.freeze_backbone = False

    logger.info("=== MobileNetV3-Small Classifier ===")
    logger.info("Data dir    : %s", data_cfg.data_dir)
    logger.info("Epochs      : %d", model_cfg.epochs)
    logger.info("Batch size  : %d", model_cfg.batch_size)
    logger.info("LR          : %s", model_cfg.learning_rate)
    logger.info("Freeze backbone: %s", model_cfg.freeze_backbone)

    try:
        loader = DatasetLoader(data_cfg)
        paths, labels = loader.load()
    except InsufficientDataError as exc:
        logger.error("Dataset error: %s", exc)
        sys.exit(1)

    splitter = DataSplitter(data_cfg)
    split_result = splitter.split(paths, labels)

    trainer = MobileNetTrainer(model_cfg)
    results = trainer.train(split_result)

    calc = MetricsCalculator()
    calc.print_report(results, "MobileNetV3-Small")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    viz = ResultVisualizer()

    cm_key = "confusion_matrix_sum" if split_result.strategy == "cv" else "confusion_matrix"
    if cm_key in results:
        viz.plot_confusion_matrix(
            results[cm_key],
            "MobileNetV3-Small — Confusion Matrix",
            str(results_dir / "confusion_matrix.png"),
        )

    if "history" in results:
        viz.plot_training_history(
            results["history"],
            str(results_dir / "training_history.png"),
        )

    logger.info("Done. Results saved to %s", results_dir)


if __name__ == "__main__":
    main()
