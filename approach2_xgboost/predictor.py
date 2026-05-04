"""Single-image inference for XGBoost with optional rotation augmentation."""
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from approach2_xgboost.feature_extractor import ImageFeatureExtractor
from approach2_xgboost.model import ModelNotTrainedError, XGBModel
from config import XGBoostConfig

logger = logging.getLogger(__name__)

_CLASS_NAMES = {0: "NO_CARD", 1: "HAS_CARD"}


class XGBPredictor:
    """Loads a trained XGBModel and runs single-image inference.

    Supports standard inference and rotation-augmented inference that
    averages predictions across four canonical orientations.
    """

    def __init__(self, model_path: str, config: XGBoostConfig) -> None:
        """Load model and initialise feature extractor.

        Args:
            model_path: Path to the saved XGBoost .json model file.
            config: XGBoostConfig instance.

        Raises:
            ModelNotTrainedError: If the model file does not exist.
        """
        if not Path(model_path).exists():
            raise ModelNotTrainedError(
                f"No trained model found at '{model_path}'. "
                "Run run_xgboost.py first."
            )
        self._config = config
        self._model = XGBModel(config)
        self._model.load(model_path)
        self._extractor = ImageFeatureExtractor(config)

    def predict(self, image_path: str) -> Dict:
        """Run single-pass inference on one image.

        Args:
            image_path: Path to the input image.

        Returns:
            Dict with keys: label (int), confidence (float), class (str).
        """
        features = self._extractor.extract(image_path)
        probs = self._model.predict_proba(features.reshape(1, -1))[0]
        label = int(np.argmax(probs))
        return {
            "label": label,
            "confidence": float(probs[label]),
            "class": _CLASS_NAMES[label],
        }

    def predict_with_rotation_augment(self, image_path: str) -> Dict:
        """Inference averaged across four canonical rotations (0°/90°/180°/270°).

        Saves a temporarily rotated version of the image to a temp path,
        extracts features, and averages probabilities.

        Args:
            image_path: Path to the input image.

        Returns:
            Dict with keys: label, confidence, class, rotation_augment (True).
        """
        import tempfile
        import os

        rotations = [0, 90, 180, 270]
        avg_probs = np.zeros(2)

        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")

        with tempfile.TemporaryDirectory() as tmpdir:
            for angle in rotations:
                rotated = img_rgb.rotate(angle, expand=True)
                tmp_path = os.path.join(tmpdir, f"rot_{angle}.jpg")
                rotated.save(tmp_path)
                features = self._extractor.extract(tmp_path)
                probs = self._model.predict_proba(features.reshape(1, -1))[0]
                avg_probs += probs

        avg_probs /= len(rotations)
        label = int(np.argmax(avg_probs))
        return {
            "label": label,
            "confidence": float(avg_probs[label]),
            "class": _CLASS_NAMES[label],
            "rotation_augment": True,
        }
