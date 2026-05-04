"""Visualization utilities for confusion matrices, training history, and comparisons."""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Generates and saves plots for model evaluation results."""

    def plot_confusion_matrix(
        self, cm: np.ndarray, title: str, save_path: str
    ) -> None:
        """Save a seaborn heatmap of the confusion matrix.

        Args:
            cm: 2×2 confusion matrix array (sklearn layout: [[TN,FP],[FN,TP]]).
            title: Plot title.
            save_path: File path to save the PNG.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        labels = ["No Card", "Has Card"]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", save_path)

    def plot_training_history(self, history: Dict, save_path: str) -> None:
        """Plot loss and accuracy curves for MobileNet training.

        Args:
            history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
            save_path: File path to save the PNG.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(history.get("train_loss", [])) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(epochs, history.get("train_loss", []), label="Train Loss", color="steelblue")
        ax1.plot(epochs, history.get("val_loss", []), label="Val Loss", color="darkorange")
        ax1.set_title("Loss per Epoch", fontsize=13)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history.get("train_acc", []), label="Train Acc", color="steelblue")
        ax2.plot(epochs, history.get("val_acc", []), label="Val Acc", color="darkorange")
        ax2.set_title("Accuracy per Epoch", fontsize=13)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("MobileNetV3 Training History", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Training history plot saved to %s", save_path)

    def plot_feature_importance(
        self, importance: Dict[str, float], save_path: str
    ) -> None:
        """Plot horizontal bar chart of top-20 XGBoost feature importances.

        Args:
            importance: Dict mapping feature_name → importance_score.
            save_path: File path to save the PNG.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        names = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(range(len(names)), scores[::-1], color="steelblue", edgecolor="white")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel("Importance Score")
        ax.set_title("Top-20 XGBoost Feature Importances", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Feature importance plot saved to %s", save_path)

    def plot_comparison(
        self, results: Dict[str, Dict], save_path: str
    ) -> None:
        """Grouped bar chart comparing approaches across all metrics.

        Args:
            results: Dict mapping approach_name → metric summary dict.
                     Each metric can be a float or a dict with 'mean'/'std'.
            save_path: File path to save the PNG.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
        approaches = list(results.keys())
        n_metrics = len(metrics)
        n_approaches = len(approaches)

        x = np.arange(n_metrics)
        width = 0.8 / n_approaches
        colors = ["steelblue", "darkorange", "seagreen", "crimson"]

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, approach in enumerate(approaches):
            means = []
            stds = []
            for metric in metrics:
                val = results[approach].get(metric, 0)
                if isinstance(val, dict):
                    means.append(val.get("mean", 0))
                    stds.append(val.get("std", 0))
                else:
                    means.append(float(val) if val is not None else 0)
                    stds.append(0)

            offset = (i - n_approaches / 2 + 0.5) * width
            ax.bar(
                x + offset,
                means,
                width,
                label=approach,
                color=colors[i % len(colors)],
                yerr=stds,
                capsize=4,
                alpha=0.85,
                edgecolor="white",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_title("Approach Comparison", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Comparison plot saved to %s", save_path)
