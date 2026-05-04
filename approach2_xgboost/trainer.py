"""XGBoost training loop with feature caching and CV/fixed-split support."""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from approach2_xgboost.feature_extractor import ImageFeatureExtractor
from approach2_xgboost.model import XGBModel
from config import XGBoostConfig
from shared.metrics import MetricsCalculator
from shared.splitter import SplitResult

logger = logging.getLogger(__name__)


class XGBTrainer:
    """Trains XGBModel for binary card/no-card classification.

    Features are extracted once and cached to disk. On subsequent runs the
    cache is reused if the image file list has not changed.
    Supports 5-Fold CV and fixed train/val/test splits.
    """

    def __init__(self, config: XGBoostConfig) -> None:
        """Initialize with XGBoost configuration.

        Args:
            config: XGBoostConfig instance.
        """
        self._config = config
        self._metrics = MetricsCalculator()

    def train(
        self,
        split_result: SplitResult,
        feature_extractor: ImageFeatureExtractor,
    ) -> Dict:
        """Extract features (or load cache) then train and evaluate.

        Args:
            split_result: SplitResult from DataSplitter.
            feature_extractor: ImageFeatureExtractor instance.

        Returns:
            Aggregated metrics dict plus feature importance.
        """
        logger.info("Starting XGBoost training — strategy: %s", split_result.strategy)

        X_all, y_all = self._get_features(split_result.all_paths, split_result.all_labels, feature_extractor)

        if split_result.strategy == "cv":
            return self._train_cv(X_all, y_all, split_result, feature_extractor)
        return self._train_fixed(X_all, y_all, split_result, feature_extractor)

    # ------------------------------------------------------------------
    # Feature caching
    # ------------------------------------------------------------------

    def _get_features(
        self,
        all_paths: List[str],
        all_labels: List[int],
        extractor: ImageFeatureExtractor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return feature matrix and labels, using disk cache when possible.

        The cache key is a hash of the sorted file path list so that any
        change in the image set triggers a full re-extraction.

        Args:
            all_paths: Ordered list of all image paths.
            all_labels: Corresponding labels.
            extractor: ImageFeatureExtractor.

        Returns:
            Tuple (X, y) as float32 / int arrays.
        """
        cache_dir = Path(self._config.feature_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._compute_cache_key(all_paths)
        key_file = cache_dir / "cache_key.txt"
        feat_file = cache_dir / "features.npy"
        label_file = cache_dir / "labels.npy"

        if (
            feat_file.exists()
            and label_file.exists()
            and key_file.exists()
            and key_file.read_text().strip() == cache_key
        ):
            logger.info("Loaded features from cache (%s)", feat_file)
            X = np.load(str(feat_file))
            y = np.load(str(label_file))
        else:
            logger.info("Extracting features from scratch (cache miss or new dataset)")
            X = extractor.extract_batch(all_paths)
            y = np.array(all_labels, dtype=np.int32)
            np.save(str(feat_file), X)
            np.save(str(label_file), y)
            key_file.write_text(cache_key)
            logger.info("Features cached to %s", cache_dir)

        return X, y

    @staticmethod
    def _compute_cache_key(paths: List[str]) -> str:
        payload = json.dumps(sorted(paths), sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Cross-validation path
    # ------------------------------------------------------------------

    def _train_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_result: SplitResult,
        extractor: ImageFeatureExtractor,
    ) -> Dict:
        fold_results: List[Dict] = []
        feature_names = extractor.get_feature_names()
        all_importances: List[Dict[str, float]] = []

        for fold_idx, (train_indices, val_indices) in enumerate(split_result.folds):
            logger.info(
                "Fold %d/%d — %d train, %d val",
                fold_idx + 1,
                split_result.n_folds,
                len(train_indices),
                len(val_indices),
            )
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[val_indices]
            y_val = y[val_indices]

            model = XGBModel(self._config)
            model.fit(X_train, y_train, X_val, y_val)

            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            metrics = self._metrics.calculate(y_val.tolist(), y_pred.tolist(), y_prob.tolist())
            fold_results.append(metrics)
            all_importances.append(model.get_feature_importance(feature_names))

        summary = self._metrics.summarize_cv(fold_results)
        summary["feature_importance"] = self._average_importances(all_importances)
        logger.info("XGBoost CV training complete")
        return summary

    # ------------------------------------------------------------------
    # Fixed split path
    # ------------------------------------------------------------------

    def _train_fixed(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        split_result: SplitResult,
        extractor: ImageFeatureExtractor,
    ) -> Dict:
        path_to_idx = {path: idx for idx, path in enumerate(split_result.all_paths)}

        train_idx = [path_to_idx[p] for p in split_result.train[0]]
        val_idx = [path_to_idx[p] for p in split_result.val[0]]
        test_idx = [path_to_idx[p] for p in split_result.test[0]]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        model = XGBModel(self._config)
        model.fit(X_train, y_train, X_val, y_val)
        model.save(self._config.model_save_path)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = self._metrics.calculate(y_test.tolist(), y_pred.tolist(), y_prob.tolist())
        feature_names = extractor.get_feature_names()
        metrics["feature_importance"] = model.get_feature_importance(feature_names)

        logger.info("XGBoost fixed-split training complete")
        return metrics

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _average_importances(importances: List[Dict[str, float]]) -> Dict[str, float]:
        """Average feature importances across CV folds.

        Args:
            importances: List of feature_name→score dicts from each fold.

        Returns:
            Averaged dict, sorted descending.
        """
        if not importances:
            return {}
        all_keys = set().union(*importances)
        avg: Dict[str, float] = {
            key: float(np.mean([d.get(key, 0.0) for d in importances]))
            for key in all_keys
        }
        return dict(sorted(avg.items(), key=lambda x: x[1], reverse=True))
