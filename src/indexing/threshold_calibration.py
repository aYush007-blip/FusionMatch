"""Bayesian Threshold Calibrator for Category-Specific Duplicate Decision Cutoffs."""

from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize_scalar


class BayesianThresholdCalibrator:
    """Per-category threshold selection via a Beta-Binomial model over
    (similarity_score, is_true_duplicate) validation pairs.
    
    Places a Beta prior over duplicate rates at candidate thresholds to maximize
    expected F1 under posterior uncertainty, ensuring robust calibration even
    on small per-category validation slices.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.thresholds_by_category: Dict[str, float] = {}

    def _expected_f1_at_threshold(
        self,
        sims: np.ndarray,
        labels: np.ndarray,
        t: float,
    ) -> float:
        """Calculates Bayesian-smoothed expected F1 score at cutoff t."""
        preds = (sims >= t).astype(int)
        tp = float(np.sum((preds == 1) & (labels == 1)))
        fp = float(np.sum((preds == 1) & (labels == 0)))
        fn = float(np.sum((preds == 0) & (labels == 1)))

        # Beta posterior mean smoothing
        precision = (tp + self.prior_alpha) / (tp + fp + self.prior_alpha + self.prior_beta)
        recall = (tp + self.prior_alpha) / (tp + fn + self.prior_alpha + self.prior_beta)

        if precision + recall == 0:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))

    def fit_category(self, sims: np.ndarray, labels: np.ndarray) -> float:
        """Finds optimal decision threshold maximizing expected F1 for a category."""
        if len(sims) == 0:
            return 0.70

        result = minimize_scalar(
            lambda t: -self._expected_f1_at_threshold(sims, labels, float(t)),
            bounds=(0.30, 0.99),
            method="bounded",
        )
        return float(np.clip(result.x, 0.30, 0.99))

    def fit(self, val_pairs_by_category: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Fits optimal thresholds for each product category and sets a global fallback.
        
        Args:
            val_pairs_by_category: Dict of {category_name: (sims_array, binary_labels_array)}
            
        Returns:
            thresholds_by_category: Dict of {category_name: calibrated_threshold}
        """
        all_sims_list = []
        all_labels_list = []

        for category, (sims, labels) in val_pairs_by_category.items():
            if len(sims) > 0:
                self.thresholds_by_category[category] = self.fit_category(sims, labels)
                all_sims_list.append(sims)
                all_labels_list.append(labels)

        # Fit global default threshold across all pooled validation pairs
        if all_sims_list:
            all_sims = np.concatenate(all_sims_list)
            all_labels = np.concatenate(all_labels_list)
            self.thresholds_by_category["__default__"] = self.fit_category(all_sims, all_labels)
        else:
            self.thresholds_by_category["__default__"] = 0.70

        return self.thresholds_by_category

    def get_threshold(self, category: Optional[str] = None) -> float:
        """Returns calibrated threshold for category or default fallback."""
        if category and category in self.thresholds_by_category:
            return self.thresholds_by_category[category]
        return self.thresholds_by_category.get("__default__", 0.70)

    def save(self, save_path: str | Path) -> Path:
        """Saves calibrated thresholds dictionary to JSON."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.thresholds_by_category, f, indent=2)
        return path

    @classmethod
    def load(cls, load_path: str | Path) -> "BayesianThresholdCalibrator":
        """Loads calibrator instance from JSON."""
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Thresholds file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        calibrator = cls()
        calibrator.thresholds_by_category = data
        return calibrator
