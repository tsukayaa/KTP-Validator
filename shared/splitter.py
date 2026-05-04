"""Dataset splitting strategy: 5-Fold Stratified CV for small datasets, fixed split otherwise."""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Unified result object for both CV and fixed-split strategies.

    For CV strategy:
        - strategy = "cv"
        - n_folds is set
        - folds is a list of (train_indices, val_indices) tuples referencing all_paths
        - all_paths and all_labels hold the full dataset

    For fixed strategy:
        - strategy = "fixed"
        - train / val / test hold (paths, labels) tuples directly
        - all_paths and all_labels still hold the full dataset (for feature caching)
    """

    strategy: str
    all_paths: List[str]
    all_labels: List[int]
    n_folds: Optional[int] = None
    folds: Optional[List[Tuple[List[int], List[int]]]] = field(default=None)
    train: Optional[Tuple[List[str], List[int]]] = None
    val: Optional[Tuple[List[str], List[int]]] = None
    test: Optional[Tuple[List[str], List[int]]] = None


class DataSplitter:
    """Selects split strategy automatically based on dataset size.

    Uses 5-Fold Stratified Cross-Validation when n_samples < threshold,
    otherwise a fixed stratified 70/20/10 train/val/test split.
    """

    def __init__(self, config: DataConfig) -> None:
        """Initialize with data configuration.

        Args:
            config: DataConfig instance with threshold, fold count, ratios, and random_state.
        """
        self._config = config

    def split(self, paths: List[str], labels: List[int]) -> SplitResult:
        """Determine and execute the appropriate split strategy.

        Args:
            paths: List of image file paths.
            labels: Corresponding integer labels (1=positive, 0=negative).

        Returns:
            SplitResult with populated fields for the chosen strategy.
        """
        n_samples = len(paths)

        if n_samples < self._config.small_dataset_threshold:
            logger.info(
                "Dataset size %d < threshold %d → using %d-Fold Stratified CV",
                n_samples,
                self._config.small_dataset_threshold,
                self._config.cv_folds,
            )
            return self._make_cv_split(paths, labels)
        else:
            logger.info(
                "Dataset size %d ≥ threshold %d → using fixed 70/20/10 split",
                n_samples,
                self._config.small_dataset_threshold,
            )
            return self._make_fixed_split(paths, labels)

    def _make_cv_split(self, paths: List[str], labels: List[int]) -> SplitResult:
        """Build a CV SplitResult using StratifiedKFold.

        Args:
            paths: All image paths.
            labels: All corresponding labels.

        Returns:
            SplitResult with strategy="cv" and populated folds list.
        """
        skf = StratifiedKFold(
            n_splits=self._config.cv_folds,
            shuffle=True,
            random_state=self._config.random_state,
        )

        labels_arr = np.array(labels)
        folds: List[Tuple[List[int], List[int]]] = []

        for train_idx, val_idx in skf.split(paths, labels_arr):
            folds.append((train_idx.tolist(), val_idx.tolist()))

        logger.info(
            "CV split ready: %d folds, ~%d train / ~%d val per fold",
            self._config.cv_folds,
            len(folds[0][0]),
            len(folds[0][1]),
        )

        return SplitResult(
            strategy="cv",
            all_paths=list(paths),
            all_labels=list(labels),
            n_folds=self._config.cv_folds,
            folds=folds,
        )

    def _make_fixed_split(self, paths: List[str], labels: List[int]) -> SplitResult:
        """Build a fixed train/val/test SplitResult.

        Args:
            paths: All image paths.
            labels: All corresponding labels.

        Returns:
            SplitResult with strategy="fixed" and populated train/val/test tuples.
        """
        test_ratio = self._config.split_ratios[2]
        val_ratio_of_remaining = self._config.split_ratios[1] / (
            self._config.split_ratios[0] + self._config.split_ratios[1]
        )

        # First: carve out test set
        trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(
            paths,
            labels,
            test_size=test_ratio,
            stratify=labels,
            random_state=self._config.random_state,
        )

        # Second: split remaining into train and val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            trainval_paths,
            trainval_labels,
            test_size=val_ratio_of_remaining,
            stratify=trainval_labels,
            random_state=self._config.random_state,
        )

        logger.info(
            "Fixed split ready: %d train / %d val / %d test",
            len(train_paths),
            len(val_paths),
            len(test_paths),
        )

        return SplitResult(
            strategy="fixed",
            all_paths=list(paths),
            all_labels=list(labels),
            train=(train_paths, train_labels),
            val=(val_paths, val_labels),
            test=(test_paths, test_labels),
        )
