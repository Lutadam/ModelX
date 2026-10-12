"""Training and scoring of the CNN+GRU (Hitesh), in the form the evaluation needs.

`make_fit_and_score` returns `fit_and_score(fit_rows, score_rows)`: it trains a
FRESH network on `fit_rows` and returns one risk probability per row of
`score_rows`. The first `sequence_minutes - 1` minutes of a block, and any
minute right after a gap, have no full window and get NaN.

The whole table lives on the GPU and batches are cut from it there, so no data
loader is needed.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from models.cnn_gru_network import RailGuardHybridNet
from models.compute import set_random_seeds
from models.sequences import find_window_end_rows, gather_windows

SCORING_BATCH_SIZE = 1024


def to_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.array(values, dtype=np.float32), device=device)


def choose_training_windows(training_rows: pd.DataFrame, sequence_minutes: int,
                            max_normal_windows: int) -> np.ndarray:
    """Row numbers that end the training windows: every risk window plus a sample of normal ones."""
    end_rows = find_window_end_rows(training_rows.index, sequence_minutes)
    window_labels = training_rows["label"].to_numpy()[end_rows]
    risk_end_rows = end_rows[window_labels == 1]
    normal_end_rows = end_rows[window_labels == 0]
    if len(normal_end_rows) > max_normal_windows:
        random_generator = np.random.default_rng(config.RANDOM_SEED)
        normal_end_rows = np.sort(random_generator.choice(normal_end_rows, max_normal_windows, replace=False))
    return np.concatenate([risk_end_rows, normal_end_rows])


def train_network(training_rows: pd.DataFrame, feature_names: list, scaler: StandardScaler,
                  device: torch.device, sequence_minutes: int, epochs: int,
                  max_normal_windows: int, verbose: bool = False) -> RailGuardHybridNet:
    """Train a fresh network on the training minutes.

    The loss is binary cross-entropy where a risk window counts
    DEEP_LEARNING_RISK_WEIGHT times more than a normal one, because risk
    windows are rare. Optimiser: AdamW with a cosine learning-rate schedule.
    """
    scaled_values = to_tensor(scaler.transform(training_rows[feature_names]), device)
    labels = to_tensor(training_rows["label"].to_numpy(), device)
    training_end_rows = torch.as_tensor(
        choose_training_windows(training_rows, sequence_minutes, max_normal_windows),
        dtype=torch.long, device=device)

    network = RailGuardHybridNet(num_features=len(feature_names)).to(device)
    optimiser = AdamW(network.parameters(), lr=config.DEEP_LEARNING_LEARNING_RATE)
    learning_rate_schedule = CosineAnnealingLR(optimiser, T_max=epochs)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.DEEP_LEARNING_RISK_WEIGHT, device=device))
    shuffle_generator = torch.Generator().manual_seed(config.RANDOM_SEED)

    for epoch in range(epochs):
        network.train()
        shuffled_order = torch.randperm(len(training_end_rows), generator=shuffle_generator).to(device)
        total_loss = 0.0
        for first in range(0, len(shuffled_order), config.DEEP_LEARNING_BATCH_SIZE):
            batch_end_rows = training_end_rows[shuffled_order[first:first + config.DEEP_LEARNING_BATCH_SIZE]]
            windows = gather_windows(scaled_values, batch_end_rows, sequence_minutes)
            optimiser.zero_grad()
            loss = loss_function(network(windows).squeeze(1), labels[batch_end_rows])
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch_end_rows)
        learning_rate_schedule.step()
        if verbose:
            print(f"  epoch {epoch + 1}/{epochs}  loss {total_loss / len(training_end_rows):.4f}", flush=True)
    return network


def score_network(network: RailGuardHybridNet, scoring_rows: pd.DataFrame, feature_names: list,
                  scaler: StandardScaler, device: torch.device, sequence_minutes: int) -> np.ndarray:
    """Risk probability per row of `scoring_rows`; NaN where there is no full window."""
    scaled_values = to_tensor(scaler.transform(scoring_rows[feature_names]), device)
    end_rows = find_window_end_rows(scoring_rows.index, sequence_minutes)
    risk_scores = np.full(len(scoring_rows), np.nan)

    network.eval()
    with torch.no_grad():
        for first in range(0, len(end_rows), SCORING_BATCH_SIZE):
            batch_end_rows = end_rows[first:first + SCORING_BATCH_SIZE]
            windows = gather_windows(
                scaled_values, torch.as_tensor(batch_end_rows, dtype=torch.long, device=device), sequence_minutes)
            risk_scores[batch_end_rows] = torch.sigmoid(network(windows)).squeeze(1).cpu().numpy()
    return risk_scores


def make_fit_and_score(feature_names: list, device: torch.device,
                       sequence_minutes: int = config.DEEP_LEARNING_SEQUENCE_MINUTES,
                       epochs: int = config.DEEP_LEARNING_EPOCHS,
                       max_normal_windows: int = config.DEEP_LEARNING_MAX_NORMAL_WINDOWS,
                       verbose: bool = False):
    """Build the `fit_and_score(fit_rows, score_rows)` function the evaluation expects.

    Args:
        feature_names: Feature columns of the minute table.
        device: From `models.compute.get_device`.
        sequence_minutes: Window length in minutes.
        epochs: Passes over the training windows.
        max_normal_windows: Normal training windows kept (all risk windows are kept).
        verbose: Print the loss after every epoch.

    Returns:
        A function that trains a fresh network and returns one risk probability
        per row of `score_rows` (NaN where there is no full window).
    """

    def fit_and_score(fit_rows: pd.DataFrame, score_rows: pd.DataFrame) -> np.ndarray:
        set_random_seeds()
        normal_features = fit_rows.loc[fit_rows["label"] == 0, feature_names]
        scaler = StandardScaler().fit(normal_features)  # scaling learned from normal running only
        network = train_network(fit_rows, feature_names, scaler, device, sequence_minutes,
                                epochs, max_normal_windows, verbose)
        return score_network(network, score_rows, feature_names, scaler, device, sequence_minutes)

    return fit_and_score
