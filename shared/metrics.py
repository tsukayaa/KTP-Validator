"""Metrics calculation and formatted reporting for binary classification."""
import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Computes and reports binary classification metrics.

    Supports single-fold evaluation, multi-fold CV aggregation,
    and formatted console reporting.
    """

    def calculate(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_prob: Optional[List[float]] = None,
    ) -> Dict:
        """Compute standard binary classification metrics.

        Args:
            y_true: Ground-truth labels (0 or 1).
            y_pred: Predicted labels (0 or 1).
            y_prob: Predicted probabilities for the positive class (optional).

        Returns:
            Dict with keys: accuracy, precision, recall, f1, confusion_matrix,
            and optionally auc_roc.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        cm = confusion_matrix(y_true_arr, y_pred_arr)

        results: Dict = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "precision": float(
                precision_score(y_true_arr, y_pred_arr, zero_division=0)
            ),
            "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
            "confusion_matrix": cm,
        }

        if y_prob is not None:
            try:
                results["auc_roc"] = float(
                    roc_auc_score(y_true_arr, np.array(y_prob))
                )
            except ValueError as exc:
                logger.warning("AUC-ROC could not be computed: %s", exc)
                results["auc_roc"] = float("nan")

        return results

    def summarize_cv(self, fold_results: List[Dict]) -> Dict:
        """Aggregate per-fold metrics into mean ± std.

        Args:
            fold_results: List of metric dicts, one per fold.

        Returns:
            Dict with same keys but values replaced by (mean, std) tuples,
            plus 'confusion_matrix_sum' as the element-wise sum across folds.
        """
        scalar_keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        summary: Dict = {}

        for key in scalar_keys:
            values = [r[key] for r in fold_results if key in r and not np.isnan(r[key])]
            if values:
                summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        cms = [r["confusion_matrix"] for r in fold_results if "confusion_matrix" in r]
        if cms:
            summary["confusion_matrix_sum"] = np.sum(cms, axis=0)

        return summary

    def print_report(self, results: Dict, approach_name: str) -> None:
        """Pretty-print evaluation results to the logger.

        Args:
            results: Output of calculate() or summarize_cv().
            approach_name: Display name for the approach (e.g. "MobileNetV3").
        """
        is_cv = "accuracy" in results and isinstance(results["accuracy"], dict)

        def fmt(key: str) -> str:
            if key not in results:
                return "N/A"
            val = results[key]
            if isinstance(val, dict):
                return f"{val['mean']:.4f} ± {val['std']:.4f}"
            return f"{val:.4f}"

        strategy = "5-Fold CV" if is_cv else "Fixed Split"

        lines = [
            "═" * 44,
            f"  {approach_name} — Evaluation Results",
            "═" * 44,
            f"  Strategy    : {strategy}",
            f"  Accuracy    : {fmt('accuracy')}",
            f"  Precision   : {fmt('precision')}",
            f"  Recall      : {fmt('recall')}",
            f"  F1 Score    : {fmt('f1')}",
            f"  AUC-ROC     : {fmt('auc_roc')}",
            "─" * 44,
        ]

        cm_key = "confusion_matrix_sum" if is_cv else "confusion_matrix"
        if cm_key in results:
            cm = results[cm_key]
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                lines.append("  Confusion Matrix (aggregated):")
                lines.append(f"    TP: {tp:<4}  FP: {fp}")
                lines.append(f"    FN: {fn:<4}  TN: {tn}")
        elif "confusion_matrix" in results:
            cm = results["confusion_matrix"]
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                lines.append("  Confusion Matrix:")
                lines.append(f"    TP: {tp:<4}  FP: {fp}")
                lines.append(f"    FN: {fn:<4}  TN: {tn}")

        lines.append("═" * 44)
        report = "\n".join(lines)
        logger.info("\n%s", report)
        print(report)
