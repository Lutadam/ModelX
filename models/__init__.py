"""Models for RailGuard AI.

One module per model, so each owner works in their own file:
    rules.py                  LPS rule and low-pressure rule (Atul)
    random_forest.py          supervised baseline (Atul)
    isolation_forest.py       unsupervised baseline (Atul)
    deep_learning_model.py    CNN+GRU training and scoring (Hitesh)
        cnn_gru_network.py    the network itself
        sequences.py          cutting the minute table into windows
        compute.py            GPU choice and random seeds

The deep learning modules are not imported here, so the baselines work without
PyTorch. Import them directly, for example
`from models.deep_learning_model import make_fit_and_score`.

Import the baselines from the package, for example `from models import train_random_forest`.
"""
from models.isolation_forest import IsolationForestDetector
from models.random_forest import predict_random_forest_alarms, train_random_forest

__all__ = ["IsolationForestDetector", "predict_random_forest_alarms", "train_random_forest"]
