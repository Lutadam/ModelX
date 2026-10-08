"""Models for RailGuard AI.

One module per model, so each owner works in their own file:
    random_forest.py     supervised baseline (Atul)
    isolation_forest.py  unsupervised baseline (Atul)
    (deep learning model: Hitesh adds his module here, for example cnn_gru.py)

Import from the package, for example `from models import train_random_forest`.
"""
from models.isolation_forest import IsolationForestDetector
from models.random_forest import predict_random_forest_alarms, train_random_forest

__all__ = ["IsolationForestDetector", "predict_random_forest_alarms", "train_random_forest"]
