"""Handcrafted, rotation-invariant CV feature extraction for card detection."""
import logging
from typing import List

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm

from config import XGBoostConfig
from shared.data_loader import InvalidImageError

logger = logging.getLogger(__name__)


class ImageFeatureExtractor:
    """Extracts a fixed-length, rotation-invariant feature vector from each image.

    All features are computed on grayscale pixels so the pipeline handles
    colour and black-and-white images identically.

    Feature groups:
        - LBP texture (uniform, rotation-invariant) at multiple radii
        - Edge/line statistics (Canny + Sobel + HoughLinesP)
        - DCT frequency-domain characteristics
        - GLCM texture properties (averaged across angles for rotation invariance)
        - Rectangle-contour shape features
        - Global statistical moments
    """

    def __init__(self, config: XGBoostConfig) -> None:
        """Initialize with XGBoost configuration.

        Args:
            config: XGBoostConfig with image_size, LBP radii, GLCM settings, etc.
        """
        self._config = config
        self._feature_names: List[str] = self._build_feature_names()
        self._logged_length = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, image_path: str) -> np.ndarray:
        """Extract a 1-D feature vector for a single image.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            np.ndarray of shape [N_features].

        Raises:
            InvalidImageError: If the image cannot be loaded.
        """
        gray = self._load_gray(image_path)
        vector = np.concatenate(
            [
                self._extract_lbp(gray),
                self._extract_edge_features(gray),
                self._extract_frequency_features(gray),
                self._extract_texture_features(gray),
                self._extract_shape_features(gray),
                self._extract_statistical_features(gray),
            ]
        )

        if not self._logged_length:
            logger.info("Feature vector length: %d", len(vector))
            self._logged_length = True

        return vector.astype(np.float32)

    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        """Extract features for all images in a list.

        Args:
            image_paths: List of image file paths.

        Returns:
            np.ndarray of shape [N_images, N_features].
        """
        vectors: List[np.ndarray] = []
        for path in tqdm(image_paths, desc="Extracting features", unit="img"):
            try:
                vectors.append(self.extract(path))
            except (InvalidImageError, Exception) as exc:
                logger.warning("Feature extraction failed for %s: %s", path, exc)
                vectors.append(np.zeros(len(self._feature_names), dtype=np.float32))

        return np.stack(vectors, axis=0)

    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names matching the feature vector.

        Returns:
            List of feature name strings.
        """
        return self._feature_names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_gray(self, image_path: str) -> np.ndarray:
        """Load an image and return a uint8 grayscale array resized to config.image_size.

        Args:
            image_path: Path to the image.

        Returns:
            np.ndarray of shape (H, W) dtype uint8.

        Raises:
            InvalidImageError: On load failure.
        """
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize(
                    (self._config.image_size, self._config.image_size),
                    Image.LANCZOS,
                )
                gray = np.array(img_resized.convert("L"), dtype=np.uint8)
            return gray
        except Exception as exc:
            raise InvalidImageError(f"Cannot load {image_path}: {exc}") from exc

    # ------------------------------------------------------------------
    # LBP — rotation-invariant texture
    # ------------------------------------------------------------------

    def _extract_lbp(self, gray: np.ndarray) -> np.ndarray:
        """LBP histograms at multiple radii.

        Uses method='uniform' which is naturally rotation-invariant.
        Produces (n_points + 2) bins per radius.

        Args:
            gray: uint8 grayscale image.

        Returns:
            Concatenated normalised histograms for all configured radii.
        """
        histograms: List[np.ndarray] = []
        for radius in self._config.lbp_radii:
            n_points = radius * self._config.lbp_n_points_factor
            lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(float)
            total = hist.sum()
            if total > 0:
                hist /= total
            histograms.append(hist)
        return np.concatenate(histograms)

    # ------------------------------------------------------------------
    # Edge / line features
    # ------------------------------------------------------------------

    def _extract_edge_features(self, gray: np.ndarray) -> np.ndarray:
        """Canny edges, Sobel gradients, and Hough line count.

        Args:
            gray: uint8 grayscale image.

        Returns:
            np.ndarray of shape (5,).
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        total_pixels = gray.size

        edge_pixel_ratio = edges.sum() / 255.0 / total_pixels

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_edge_magnitude = float(magnitude.mean())

        h, w = gray.shape
        center_crop = edges[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        edge_density_center = center_crop.sum() / 255.0 / max(center_crop.size, 1)

        lower_half = edges[h // 2 :, :]
        edge_density_lower_half = lower_half.sum() / 255.0 / max(lower_half.size, 1)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
        )
        num_strong_lines = 0 if lines is None else len(lines)

        return np.array(
            [
                edge_pixel_ratio,
                mean_edge_magnitude / 255.0,
                edge_density_center,
                edge_density_lower_half,
                min(num_strong_lines, 500) / 500.0,  # normalise
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # DCT frequency features
    # ------------------------------------------------------------------

    def _extract_frequency_features(self, gray: np.ndarray) -> np.ndarray:
        """2-D DCT frequency-domain signature.

        Documents/cards with printed text have distinctive high-frequency
        patterns compared to plain faces or backgrounds.

        Args:
            gray: uint8 grayscale image.

        Returns:
            np.ndarray of shape (5,).
        """
        gray_f = gray.astype(float)
        dct = dctn(gray_f, norm="ortho")
        dct_sq = dct**2
        total_energy = dct_sq.sum() + 1e-10

        h, w = dct.shape
        low_energy = dct_sq[: h // 4, : w // 4].sum()
        high_energy = dct_sq[h // 2 :, w // 2 :].sum()

        low_freq_energy_ratio = float(low_energy / total_energy)
        high_freq_energy_ratio = float(high_energy / total_energy)

        flat = dct_sq.ravel()
        flat = flat / (flat.sum() + 1e-10)
        spectral_entropy = float(-np.sum(flat * np.log(flat + 1e-10)))
        max_entropy = np.log(len(flat) + 1e-10)
        spectral_entropy_norm = spectral_entropy / max_entropy

        mid_energy = float(
            dct_sq[h // 4 : h // 2, w // 4 : w // 2].sum() / total_energy
        )

        return np.array(
            [
                low_freq_energy_ratio,
                spectral_entropy_norm,
                high_freq_energy_ratio,
                mid_energy,
                float(np.log1p(np.abs(dct[0, 0]))),  # DC component magnitude
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # GLCM texture
    # ------------------------------------------------------------------

    def _extract_texture_features(self, gray: np.ndarray) -> np.ndarray:
        """GLCM texture properties averaged across angles for rotation invariance.

        Args:
            gray: uint8 grayscale image.

        Returns:
            np.ndarray of shape (n_distances * 5,).
        """
        # Quantise to 64 levels for memory efficiency
        gray_q = (gray // 4).astype(np.uint8)

        angles = np.array(self._config.glcm_angles)
        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
        features: List[float] = []

        for dist in self._config.glcm_distances:
            glcm = graycomatrix(
                gray_q,
                distances=[dist],
                angles=angles,
                levels=64,
                symmetric=True,
                normed=True,
            )
            for prop in props:
                values = graycoprops(glcm, prop)[0]  # shape: (n_angles,)
                features.append(float(values.mean()))  # average across angles

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Contour / shape features
    # ------------------------------------------------------------------

    def _extract_shape_features(self, gray: np.ndarray) -> np.ndarray:
        """Rectangle-contour features — cards have strong rectangular outlines.

        Args:
            gray: uint8 grayscale image.

        Returns:
            np.ndarray of shape (5,).
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        image_area = float(gray.shape[0] * gray.shape[1])
        num_rectangles = 0
        largest_rect_area = 0.0
        largest_rect_aspect = 0.0
        largest_rect_solidity = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < image_area * 0.01:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                num_rectangles += 1
                if area > largest_rect_area:
                    largest_rect_area = area
                    _, (w_box, h_box), _ = cv2.minAreaRect(cnt)
                    if h_box > 0:
                        largest_rect_aspect = float(max(w_box, h_box) / (min(w_box, h_box) + 1e-5))
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    largest_rect_solidity = float(area / (hull_area + 1e-5))

        return np.array(
            [
                min(num_rectangles, 20) / 20.0,
                largest_rect_area / (image_area + 1e-5),
                min(largest_rect_aspect, 10.0) / 10.0,
                min(largest_rect_solidity, 1.0),
                1.0 if largest_rect_area / (image_area + 1e-5) > 0.05 else 0.0,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Statistical moments
    # ------------------------------------------------------------------

    def _extract_statistical_features(self, gray: np.ndarray) -> np.ndarray:
        """Global pixel intensity statistics.

        Args:
            gray: uint8 grayscale image.

        Returns:
            np.ndarray of shape (6,).
        """
        flat = gray.ravel().astype(float)
        hist, _ = np.histogram(flat, bins=64, range=(0, 256), density=True)
        hist_prob = hist + 1e-10
        hist_prob /= hist_prob.sum()
        entropy = float(-np.sum(hist_prob * np.log(hist_prob)))

        return np.array(
            [
                float(flat.mean()) / 255.0,
                float(flat.std()) / 255.0,
                float(skew(flat)),
                float(np.clip(kurtosis(flat), -10, 10)),
                float(entropy) / np.log(64),
                float(flat.max() - flat.min()) / 255.0,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Feature name builder (for interpretability)
    # ------------------------------------------------------------------

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []

        for radius in self._config.lbp_radii:
            n_points = radius * self._config.lbp_n_points_factor
            n_bins = n_points + 2
            for b in range(n_bins):
                names.append(f"lbp_r{radius}_bin{b}")

        names += [
            "edge_pixel_ratio",
            "mean_edge_magnitude",
            "edge_density_center",
            "edge_density_lower_half",
            "num_strong_lines",
        ]

        names += [
            "dct_low_freq_energy",
            "dct_spectral_entropy",
            "dct_high_freq_energy",
            "dct_mid_energy",
            "dct_dc_magnitude",
        ]

        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
        for dist in self._config.glcm_distances:
            for prop in props:
                names.append(f"glcm_d{dist}_{prop}")

        names += [
            "shape_num_rectangles",
            "shape_largest_rect_area_ratio",
            "shape_largest_rect_aspect",
            "shape_rect_solidity",
            "shape_has_rect_flag",
        ]

        names += [
            "stat_mean",
            "stat_std",
            "stat_skewness",
            "stat_kurtosis",
            "stat_entropy",
            "stat_intensity_range",
        ]

        return names
