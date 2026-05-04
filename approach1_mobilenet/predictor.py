"""Single-image inference for MobileNetV3-Small with optional TTA."""
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image

from approach1_mobilenet.augmentation import AugmentationPipeline
from approach1_mobilenet.model import MobileNetClassifier, ModelNotTrainedError
from config import MobileNetConfig

logger = logging.getLogger(__name__)

_CLASS_NAMES = {0: "NO_CARD", 1: "HAS_CARD"}


class MobileNetPredictor:
    """Loads a trained MobileNetV3-Small and runs single-image inference.

    Supports standard inference and Test-Time Augmentation (TTA) over
    four canonical rotations (0°, 90°, 180°, 270°) for robust handling
    of rotated inputs.
    """

    def __init__(self, model_path: str, config: MobileNetConfig) -> None:
        """Load model weights from disk.

        Args:
            model_path: Path to the saved .pth weight file.
            config: MobileNetConfig instance.

        Raises:
            ModelNotTrainedError: If the weight file does not exist.
        """
        if not Path(model_path).exists():
            raise ModelNotTrainedError(
                f"No trained model found at '{model_path}'. "
                "Run run_mobilenet.py first."
            )
        self._config = config
        self._model = MobileNetClassifier(config)
        self._model.load(model_path)
        self._model.eval()
        self._val_transform = AugmentationPipeline("val", config)

    def predict(self, image_path: str) -> Dict:
        """Run single-pass inference on one image.

        Args:
            image_path: Path to the input image.

        Returns:
            Dict with keys: label (int), confidence (float), class (str).
        """
        tensor = self._load_tensor(image_path)
        with torch.no_grad():
            logits = self._model(tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=1).squeeze(0)

        label = int(probs.argmax().item())
        confidence = float(probs[label].item())
        return {
            "label": label,
            "confidence": confidence,
            "class": _CLASS_NAMES[label],
        }

    def predict_with_tta(self, image_path: str) -> Dict:
        """Inference with Test-Time Augmentation over 4 rotations.

        Rotates the image at 0°, 90°, 180°, and 270°, runs inference for
        each, then averages the softmax probabilities.

        Args:
            image_path: Path to the input image.

        Returns:
            Dict with keys: label, confidence, class, tta_applied (True).
        """
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")

        avg_probs = torch.zeros(2)
        for angle in self._config.tta_rotations:
            rotated = img_rgb.rotate(angle, expand=True)
            tensor = self._val_transform(rotated)
            with torch.no_grad():
                logits = self._model(tensor.unsqueeze(0))
                probs = F.softmax(logits, dim=1).squeeze(0)
            avg_probs += probs

        avg_probs /= len(self._config.tta_rotations)
        label = int(avg_probs.argmax().item())
        confidence = float(avg_probs[label].item())

        return {
            "label": label,
            "confidence": confidence,
            "class": _CLASS_NAMES[label],
            "tta_applied": True,
        }

    def _load_tensor(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as img:
            return self._val_transform(img.convert("RGB"))
