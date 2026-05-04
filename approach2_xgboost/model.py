"""XGBoost binary classifier wrapper with save/load and feature importance."""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb

from config import XGBoostConfig

logger = logging.getLogger(__name__)


class ModelNotTrainedError(Exception):
    """Raised when prediction is attempted before a model has been loaded/trained."""


class XGBModel:
    """Wraps XGBClassifier for binary card/no-card classification.

    Uses CPU-optimised 'hist' tree method. Class imbalance is handled via
    scale_pos_weight computed from training labels before each fit call.
    """

    def __init__(self, config: XGBoostConfig) -> None:
        """Initialize XGBClassifier with config hyperparameters.

        Args:
            config: XGBoostConfig instance.
        """
        self._config = config
        self._model = xgb.XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=config.random_state,
            verbosity=0,
        )
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the XGBoost model with optional early stopping.

        Computes scale_pos_weight from y_train to handle class imbalance.

        Args:
            X_train: Feature matrix, shape [n_train, n_features].
            y_train: Integer labels, shape [n_train].
            X_val: Validation features for early stopping (optional).
            y_val: Validation labels for early stopping (optional).
        """
        counts = np.bincount(y_train.astype(int), minlength=2)
        if counts[1] > 0:
            scale_pos_weight = float(counts[0]) / float(counts[1])
        else:
            scale_pos_weight = 1.0
        self._model.set_params(scale_pos_weight=scale_pos_weight)
        logger.info(
            "Training XGBoost — pos: %d, neg: %d, scale_pos_weight: %.3f",
            counts[1],
            counts[0],
            scale_pos_weight,
        )

        fit_kwargs: Dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
            fit_kwargs["callbacks"] = [xgb.callback.EarlyStopping(rounds=20, save_best=True)]

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._fitted = True
        logger.info("XGBoost training complete — best iteration: %s", self._model.best_iteration)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels.

        Args:
            X: Feature matrix, shape [n_samples, n_features].

        Returns:
            Integer label array of shape [n_samples].

        Raises:
            ModelNotTrainedError: If fit() has not been called.
        """
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities.

        Args:
            X: Feature matrix, shape [n_samples, n_features].

        Returns:
            Float array of shape [n_samples, 2].

        Raises:
            ModelNotTrainedError: If fit() has not been called.
        """
        self._check_fitted()
        return self._model.predict_proba(X)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Return a dict of feature_name → importance, sorted descending.

        Args:
            feature_names: Ordered list of feature names.

        Returns:
            Dict sorted by importance score.

        Raises:
            ModelNotTrainedError: If fit() has not been called.
        """
        self._check_fitted()
        importances = self._model.feature_importances_
        paired = {name: float(imp) for name, imp in zip(feature_names, importances)}
        return dict(sorted(paired.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        """Save the trained model to disk in XGBoost JSON format.

        Args:
            path: File path (should end in .json).

        Raises:
            ModelNotTrainedError: If fit() has not been called.
        """
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(path)
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved model from disk.

        Args:
            path: Path to the .json model file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"XGBoost model not found: {path}")
        self._model.load_model(path)
        self._fitted = True
        logger.info("XGBoost model loaded from %s", path)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise ModelNotTrainedError(
                "Model has not been trained. Call fit() or load() first."
            )
