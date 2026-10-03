"""Random Forest baseline."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_FOREST_ALARM_PROBABILITY, RANDOM_FOREST_TREES, RANDOM_SEED


def train_random_forest(features_training: pd.DataFrame, labels_training: pd.Series) -> RandomForestClassifier:
    """Fit a class-balanced Random Forest on the training minutes.

    Args:
        features_training: Feature rows with no NaN.
        labels_training: 1 for risk, 0 for normal, aligned to the features.

    Returns:
        The fitted classifier. The random seed is fixed from config.
    """
    forest = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_TREES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return forest.fit(features_training, labels_training.astype(int))


def predict_alarms(forest: RandomForestClassifier, features: pd.DataFrame) -> pd.Series:
    """Raise an alarm when the predicted risk probability reaches the config threshold.

    Args:
        forest: A fitted classifier from `train_random_forest`.
        features: Feature rows with no NaN.

    Returns:
        Boolean alarm series aligned to the features.
    """
    risk_probability = forest.predict_proba(features)[:, list(forest.classes_).index(1)]
    return pd.Series(risk_probability >= RANDOM_FOREST_ALARM_PROBABILITY, index=features.index)
