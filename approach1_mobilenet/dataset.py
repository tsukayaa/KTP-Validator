"""PyTorch Dataset for KTP binary classification with optional epoch-repeat oversampling."""
import logging
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from approach1_mobilenet.augmentation import AugmentationPipeline

logger = logging.getLogger(__name__)


class KTPDataset(Dataset):
    """Loads images from disk, converts to RGB, applies transforms.

    Supports a repeat_factor to artificially inflate the dataset length
    so each epoch sees more augmentation diversity on small datasets.
    Grayscale images are converted to 3-channel RGB transparently.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: AugmentationPipeline,
        repeat_factor: int = 1,
    ) -> None:
        """Initialize the dataset.

        Args:
            image_paths: List of absolute image file paths.
            labels: Corresponding integer labels (1=has card, 0=no card).
            transform: AugmentationPipeline to apply on each image.
            repeat_factor: Virtual epoch multiplier. Effective length =
                len(image_paths) * repeat_factor.
        """
        if len(image_paths) != len(labels):
            raise ValueError(
                f"image_paths length ({len(image_paths)}) != labels length ({len(labels)})"
            )
        self._paths = image_paths
        self._labels = labels
        self._transform = transform
        self._repeat_factor = max(1, repeat_factor)
        self._n_real = len(image_paths)

    def __len__(self) -> int:
        return self._n_real * self._repeat_factor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and transform one image.

        Args:
            idx: Virtual index (wraps around the real dataset).

        Returns:
            Tuple of (transformed_tensor, label).
        """
        real_idx = idx % self._n_real
        path = self._paths[real_idx]
        label = self._labels[real_idx]

        # Try loading; on failure, fall back to the next valid image
        for attempt in range(self._n_real):
            try:
                with Image.open(self._paths[(real_idx + attempt) % self._n_real]) as img:
                    tensor = self._transform(img.convert("RGB"))
                if attempt > 0:
                    logger.warning(
                        "Failed to load %s; using replacement at offset +%d",
                        path,
                        attempt,
                    )
                    label = self._labels[(real_idx + attempt) % self._n_real]
                return tensor, label
            except Exception as exc:
                logger.warning("Could not load image %s: %s", path, exc)

        raise RuntimeError(f"No loadable image found near index {idx}")
