"""Isolation Forest baseline (unsupervised: learns what normal looks like, flags the unusual)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from alerts import count_false_alerts, operating_hours_of_normal_minutes
from config import ISOLATION_FOREST_TREES, RANDOM_SEED


class IsolationForestDetector:
    """Alarm when a minute looks unlike normal running.

    The forest is trained on the normal training minutes only, so it never
    sees a risk window. The anomaly score cut-off is set on the training
    folds: the most sensitive cut-off whose false alerts per operating hour
    stay within the budget. It is never set on the held-out fold.
    """

    def __init__(self, false_alerts_per_operating_hour_budget: float):
        """Create an unfitted detector.

        Args:
            false_alerts_per_operating_hour_budget: Most false alerts allowed
                per operating hour, for example 1 / 100.
        """
        self.false_alerts_per_operating_hour_budget = false_alerts_per_operating_hour_budget
        self.forest = None
        self.score_cutoff = None

    def anomaly_scores(self, features: pd.DataFrame) -> pd.Series:
        """Score each minute: higher means more unusual.

        Args:
            features: Feature rows with no NaN.

        Returns:
            Anomaly score per row, aligned to the features.
        """
        return pd.Series(-self.forest.score_samples(features), index=features.index)

    def fit(self, features_training: pd.DataFrame, labels_training: pd.Series,
            operating_training: pd.Series) -> "IsolationForestDetector":
        """Train on the normal training minutes, then set the score cut-off.

        Candidate cut-offs are tried from the least sensitive (no alarms)
        downwards. The search stops at the first one that breaks the budget,
        and the one before it is kept.

        Args:
            features_training: Feature rows with no NaN, training period only.
            labels_training: Labels from `make_labels`, aligned to the features.
            operating_training: Output of `operating_per_minute`, aligned too.

        Returns:
            The fitted detector (self).
        """
        is_normal = (labels_training == 0).fillna(False).to_numpy(dtype=bool)
        normal_operating_hours = operating_hours_of_normal_minutes(labels_training, operating_training)
        if normal_operating_hours == 0:
            raise ValueError("No operating hours in the normal training minutes.")

        self.forest = IsolationForest(
            n_estimators=ISOLATION_FOREST_TREES,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ).fit(features_training[is_normal])

        training_scores = self.anomaly_scores(features_training)
        normal_scores = training_scores[is_normal]
        candidate_cutoffs = np.unique(normal_scores.quantile(np.linspace(0.5, 1.0, 501)))[::-1]
        # The highest candidate equals the largest normal score, so nothing is above it: no alarms.
        self.score_cutoff = candidate_cutoffs[0]
        for candidate_cutoff in candidate_cutoffs[1:]:
            alarm = training_scores > candidate_cutoff
            false_alerts = count_false_alerts(alarm, labels_training)
            if false_alerts / normal_operating_hours > self.false_alerts_per_operating_hour_budget:
                break
            self.score_cutoff = candidate_cutoff
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return True for minutes whose anomaly score is above the cut-off.

        Args:
            features: Feature rows with no NaN.

        Returns:
            Boolean alarm series aligned to the features.
        """
        if self.score_cutoff is None:
            raise RuntimeError("Call fit() before predict().")
        return self.anomaly_scores(features) > self.score_cutoff
