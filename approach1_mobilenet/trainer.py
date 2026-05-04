"""Training loop for MobileNetV3-Small with early stopping and CV support."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from approach1_mobilenet.augmentation import AugmentationPipeline
from approach1_mobilenet.dataset import KTPDataset
from approach1_mobilenet.model import MobileNetClassifier
from config import MobileNetConfig
from shared.metrics import MetricsCalculator
from shared.splitter import SplitResult

logger = logging.getLogger(__name__)


class MobileNetTrainer:
    """Trains MobileNetV3-Small for binary card/no-card classification.

    Supports 5-Fold CV (for small datasets) and fixed train/val/test splits.
    Applies class-weighted loss when the dataset is imbalanced.
    """

    def __init__(self, config: MobileNetConfig) -> None:
        """Initialize with MobileNet configuration.

        Args:
            config: MobileNetConfig instance.
        """
        self._config = config
        self._metrics = MetricsCalculator()

    def train(self, split_result: SplitResult) -> Dict:
        """Train and evaluate the model.

        Args:
            split_result: SplitResult from DataSplitter.

        Returns:
            Aggregated metrics dict (mean/std for CV, scalar for fixed split).
        """
        logger.info(
            "Starting MobileNet training — strategy: %s", split_result.strategy
        )
        if split_result.strategy == "cv":
            return self._train_cv(split_result)
        return self._train_fixed(split_result)

    # ------------------------------------------------------------------
    # Cross-validation path
    # ------------------------------------------------------------------

    def _train_cv(self, split_result: SplitResult) -> Dict:
        fold_results: List[Dict] = []

        for fold_idx, (train_indices, val_indices) in enumerate(split_result.folds):
            logger.info(
                "Fold %d/%d — %d train, %d val",
                fold_idx + 1,
                split_result.n_folds,
                len(train_indices),
                len(val_indices),
            )
            train_paths = [split_result.all_paths[i] for i in train_indices]
            train_labels = [split_result.all_labels[i] for i in train_indices]
            val_paths = [split_result.all_paths[i] for i in val_indices]
            val_labels = [split_result.all_labels[i] for i in val_indices]

            model = MobileNetClassifier(self._config)

            train_ds = KTPDataset(
                train_paths,
                train_labels,
                AugmentationPipeline("train", self._config),
                repeat_factor=self._config.repeat_factor,
            )
            val_ds = KTPDataset(
                val_paths,
                val_labels,
                AugmentationPipeline("val", self._config),
                repeat_factor=1,
            )

            fold_metrics, _ = self._train_fold(
                model, train_ds, val_ds, train_labels, fold_idx + 1
            )
            fold_results.append(fold_metrics)

        summary = self._metrics.summarize_cv(fold_results)
        logger.info("CV training complete")
        return summary

    # ------------------------------------------------------------------
    # Fixed split path
    # ------------------------------------------------------------------

    def _train_fixed(self, split_result: SplitResult) -> Dict:
        train_paths, train_labels = split_result.train
        val_paths, val_labels = split_result.val
        test_paths, test_labels = split_result.test

        model = MobileNetClassifier(self._config)

        train_ds = KTPDataset(
            train_paths,
            train_labels,
            AugmentationPipeline("train", self._config),
            repeat_factor=self._config.repeat_factor,
        )
        val_ds = KTPDataset(
            val_paths,
            val_labels,
            AugmentationPipeline("val", self._config),
        )
        test_ds = KTPDataset(
            test_paths,
            test_labels,
            AugmentationPipeline("val", self._config),
        )

        _, history = self._train_fold(model, train_ds, val_ds, train_labels, fold_num=None)

        save_path = str(Path(self._config.model_save_dir) / "best_model.pth")
        model.save(save_path)

        test_metrics = self._evaluate(model, test_ds)
        logger.info("Test set evaluation complete")

        return {"history": history, **test_metrics}

    # ------------------------------------------------------------------
    # Core fold training
    # ------------------------------------------------------------------

    def _train_fold(
        self,
        model: MobileNetClassifier,
        train_dataset: KTPDataset,
        val_dataset: KTPDataset,
        train_labels: List[int],
        fold_num: Optional[int],
    ) -> Tuple[Dict, Dict]:
        """Run the full training loop for one fold or fixed split.

        Args:
            model: Fresh MobileNetClassifier instance.
            train_dataset: Training KTPDataset.
            val_dataset: Validation KTPDataset.
            train_labels: Training labels (used for class-weight computation).
            fold_num: Display fold number, or None for fixed-split runs.

        Returns:
            Tuple of (best_val_metrics, training_history).
        """
        # Class weighting
        counts = np.bincount(train_labels, minlength=2).astype(float)
        if counts[0] == 0 or counts[1] == 0:
            class_weights = torch.tensor([1.0, 1.0])
        else:
            total = counts.sum()
            weights = total / (2.0 * counts)
            class_weights = torch.tensor(weights, dtype=torch.float)
        logger.info(
            "Class weights — no-card: %.3f, has-card: %.3f",
            class_weights[0].item(),
            class_weights[1].item(),
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(
            model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5, verbose=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        best_metrics: Dict = {}

        prefix = f"Fold {fold_num} " if fold_num else ""

        for epoch in range(1, self._config.epochs + 1):
            # Training step
            model.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for images, labels_batch in train_loader:
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * len(labels_batch)
                t_correct += (logits.argmax(1) == labels_batch).sum().item()
                t_total += len(labels_batch)

            train_loss = t_loss / max(t_total, 1)
            train_acc = t_correct / max(t_total, 1)

            # Validation step
            val_metrics = self._evaluate(model, val_dataset)
            val_loss = self._compute_val_loss(model, val_loader, criterion)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_metrics["accuracy"])
            history["lr"].append(current_lr)

            logger.info(
                "%sepoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
                "train_acc=%.4f  val_acc=%.4f  lr=%.2e",
                prefix,
                epoch,
                self._config.epochs,
                train_loss,
                val_loss,
                train_acc,
                val_metrics["accuracy"],
                current_lr,
            )

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_metrics = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._config.patience:
                    logger.info(
                        "%sEarly stopping at epoch %d (patience=%d)",
                        prefix,
                        epoch,
                        self._config.patience,
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return best_metrics, history

    def _compute_val_loss(
        self, model: MobileNetClassifier, loader: DataLoader, criterion: nn.Module
    ) -> float:
        model.eval()
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for images, labels_batch in loader:
                logits = model(images)
                loss = criterion(logits, labels_batch)
                total_loss += loss.item() * len(labels_batch)
                total += len(labels_batch)
        return total_loss / max(total, 1)

    def _evaluate(self, model: MobileNetClassifier, dataset: KTPDataset) -> Dict:
        """Run model over an entire dataset and return metrics.

        Args:
            model: MobileNetClassifier.
            dataset: KTPDataset to evaluate.

        Returns:
            Metrics dict from MetricsCalculator.
        """
        loader = DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        model.eval()
        all_preds: List[int] = []
        all_probs: List[float] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for images, labels_batch in loader:
                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())
                all_labels.extend(labels_batch.tolist())

        return MetricsCalculator().calculate(all_labels, all_preds, all_probs)
