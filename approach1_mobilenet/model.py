"""MobileNetV3-Small transfer-learning model for binary card detection."""
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from config import MobileNetConfig

logger = logging.getLogger(__name__)


class ModelNotTrainedError(Exception):
    """Raised when inference is attempted before a model has been loaded/trained."""


class MobileNetClassifier(nn.Module):
    """Wraps MobileNetV3-Small with a custom binary classification head.

    Backbone features are optionally frozen; only the classifier head is
    trained by default, making this efficient on small datasets.
    CPU-only — no CUDA code anywhere.
    """

    def __init__(self, config: MobileNetConfig) -> None:
        """Load pretrained backbone and replace classifier head.

        Args:
            config: MobileNetConfig with dropout, freeze_backbone flag, etc.
        """
        super().__init__()
        self._config = config
        self.device = torch.device("cpu")

        self.backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )

        if config.freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            logger.info("Backbone features frozen — only classifier head will train")

        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 2),
        )

        self.to(self.device)
        logger.info(
            "MobileNetClassifier ready — trainable params: %d", self.get_trainable_params()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, 2).
        """
        return self.backbone(x)

    def get_trainable_params(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of parameters with requires_grad=True.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Persist model weights to disk.

        Args:
            path: File path for the .pth file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("Model weights saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from disk.

        Args:
            path: File path to the .pth file.

        Raises:
            FileNotFoundError: If the weight file does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model weights not found: {path}")
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()
        logger.info("Model weights loaded from %s", path)
