"""Augmentation pipelines for MobileNet training and validation."""
import logging
from typing import Callable

import torch
from torchvision import transforms

from config import MobileNetConfig

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class AugmentationPipeline:
    """Builds torchvision transform pipelines for 'train' and 'val' modes.

    Training applies heavy spatial/colour augmentation to handle rotated
    inputs, B&W images, and varying card sizes.
    Validation applies only deterministic resize + crop + normalise.
    """

    def __init__(self, mode: str, config: MobileNetConfig) -> None:
        """Initialize the pipeline.

        Args:
            mode: Either 'train' or 'val'.
            config: MobileNetConfig with image_size and augmentation params.

        Raises:
            ValueError: If mode is not 'train' or 'val'.
        """
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got '{mode}'")

        self._mode = mode
        self._config = config
        self._transform: Callable = self._build_train() if mode == "train" else self._build_val()
        logger.debug("AugmentationPipeline built for mode=%s", mode)

    def __call__(self, image) -> torch.Tensor:
        """Apply the transform pipeline to a PIL image.

        Args:
            image: PIL.Image object.

        Returns:
            Normalised torch.Tensor of shape (3, H, W).
        """
        return self._transform(image)

    def _build_train(self) -> Callable:
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=360),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomGrayscale(p=self._config.grayscale_augment_prob),
                transforms.RandomResizedCrop(
                    self._config.image_size, scale=(0.6, 1.0)
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def _build_val(self) -> Callable:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self._config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
