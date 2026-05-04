"""
Central configuration for all hyperparameters, paths, and thresholds.
No magic numbers anywhere else — import from here.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for dataset loading and splitting."""

    data_dir: str = "./data"
    positives_dir: str = "positives"
    negatives_dir: str = "negatives"
    valid_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    small_dataset_threshold: int = 200
    cv_folds: int = 5
    split_ratios: tuple = (0.70, 0.20, 0.10)  # train, val, test
    random_state: int = 42


@dataclass
class MobileNetConfig:
    """Configuration for MobileNetV3-Small classifier."""

    image_size: int = 224
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout: float = 0.5
    patience: int = 7
    freeze_backbone: bool = True
    repeat_factor: int = 10
    tta_rotations: list = field(default_factory=lambda: [0, 90, 180, 270])
    grayscale_augment_prob: float = 0.3
    model_save_dir: str = "./models/mobilenet"


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost classifier with handcrafted CV features."""

    image_size: int = 256
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    lbp_radii: list = field(default_factory=lambda: [1, 2, 3])
    lbp_n_points_factor: int = 8
    glcm_distances: list = field(default_factory=lambda: [1, 3, 5])
    glcm_angles: list = field(default_factory=lambda: [0, 0.785, 1.571, 2.356])
    model_save_path: str = "./models/xgboost/model.json"
    feature_cache_dir: str = "./cache"
