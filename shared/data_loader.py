"""Dataset loading with image validation and class distribution reporting."""
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from config import DataConfig

logger = logging.getLogger(__name__)


class InvalidImageError(Exception):
    """Raised when an image file cannot be read or is corrupt."""


class InsufficientDataError(Exception):
    """Raised when there are not enough samples for training."""


class DatasetLoader:
    """Scans dataset directories and returns validated image paths with labels.

    Labels: 1 = positive (image contains an identity card), 0 = negative.
    """

    def __init__(self, config: DataConfig) -> None:
        """Initialize with data configuration.

        Args:
            config: DataConfig instance with directory paths and valid extensions.
        """
        self._config = config
        self._paths: List[str] = []
        self._labels: List[int] = []

    def load(self) -> Tuple[List[str], List[int]]:
        """Scan directories, validate files, and return (paths, labels).

        Returns:
            Tuple of (image_paths, labels) where label 1=has card, 0=no card.

        Raises:
            InsufficientDataError: If either directory is missing or empty.
        """
        pos_dir = Path(self._config.data_dir) / self._config.positives_dir
        neg_dir = Path(self._config.data_dir) / self._config.negatives_dir

        if not pos_dir.exists():
            raise InsufficientDataError(f"Positives directory not found: {pos_dir}")
        if not neg_dir.exists():
            raise InsufficientDataError(f"Negatives directory not found: {neg_dir}")

        pos_paths = self._scan_directory(pos_dir)
        neg_paths = self._scan_directory(neg_dir)

        if not pos_paths:
            raise InsufficientDataError(f"No valid images found in {pos_dir}")
        if not neg_paths:
            raise InsufficientDataError(f"No valid images found in {neg_dir}")

        paths = pos_paths + neg_paths
        labels = [1] * len(pos_paths) + [0] * len(neg_paths)

        logger.info(
            "Dataset loaded: %d positives, %d negatives, %d total",
            len(pos_paths),
            len(neg_paths),
            len(paths),
        )

        ratio = len(pos_paths) / len(neg_paths) if neg_paths else float("inf")
        if ratio > 3.0 or ratio < 0.33:
            logger.warning(
                "Class imbalance detected (pos:neg = %.2f:1). "
                "Class weighting will be applied during training.",
                ratio,
            )

        self._paths = paths
        self._labels = labels
        return paths, labels

    def _scan_directory(self, directory: Path) -> List[str]:
        """Scan a directory for valid image files.

        Args:
            directory: Path to the directory to scan.

        Returns:
            List of valid image file paths as strings.
        """
        valid_paths: List[str] = []
        skipped = 0

        for file_path in sorted(directory.iterdir()):
            if file_path.suffix.lower() in self._config.valid_extensions:
                if self._validate_image(str(file_path)):
                    valid_paths.append(str(file_path))
                else:
                    skipped += 1

        if skipped > 0:
            logger.warning(
                "Skipped %d corrupt/unreadable files in %s", skipped, directory
            )

        logger.info("Found %d valid images in %s", len(valid_paths), directory)
        return valid_paths

    def _validate_image(self, path: str) -> bool:
        """Attempt to load image pixel data to confirm it is readable.

        Args:
            path: Absolute path to the image file.

        Returns:
            True if the image is valid, False if corrupt or unreadable.
        """
        try:
            with Image.open(path) as img:
                img.load()
            return True
        except Exception as exc:
            logger.warning("Corrupt or unreadable image (skipped): %s — %s", path, exc)
            return False

    def get_class_distribution(self) -> Dict[str, int]:
        """Return a dict with class counts.

        Returns:
            Dict with keys 'positives', 'negatives', 'total'.
        """
        if not self._labels:
            self.load()
        positives = sum(1 for label in self._labels if label == 1)
        negatives = sum(1 for label in self._labels if label == 0)
        return {"positives": positives, "negatives": negatives, "total": len(self._labels)}
