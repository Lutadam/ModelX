"""The 1D-CNN + GRU network (architecture from Hitesh's notebook)."""
import torch
import torch.nn as nn

import config


class TemporalConvNetBlock(nn.Module):
    """Two convolutions over time, then MaxPool: halves the sequence length.

    The convolutions smooth sensor noise and pick up short pressure patterns.
    """

    def __init__(self, input_features: int, channels: int, dropout: float = config.DEEP_LEARNING_DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()
        self.pool = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """(batch, input_features, minutes) -> (batch, channels, minutes // 2)."""
        first_layer = self.dropout1(self.act1(self.bn1(self.conv1(sequences))))
        return self.dropout2(self.pool(self.act2(self.bn2(self.conv2(first_layer)))))


class RailGuardHybridNet(nn.Module):
    """Convolutions find short patterns, the GRU follows them over hours, a small head gives one risk logit."""

    def __init__(self, num_features: int,
                 cnn_channels: int = config.DEEP_LEARNING_CNN_CHANNELS,
                 gru_hidden_size: int = config.DEEP_LEARNING_GRU_HIDDEN_SIZE,
                 gru_layers: int = config.DEEP_LEARNING_GRU_LAYERS,
                 dropout: float = config.DEEP_LEARNING_DROPOUT):
        super().__init__()
        self.cnn = TemporalConvNetBlock(num_features, cnn_channels, dropout)
        self.gru = nn.GRU(cnn_channels, gru_hidden_size, num_layers=gru_layers, batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """(batch, minutes, features) -> risk logits (batch, 1). Apply a sigmoid for a probability."""
        convolved = self.cnn(sequences.transpose(1, 2)).transpose(1, 2)
        _, last_hidden_states = self.gru(convolved)
        return self.classifier(last_hidden_states[-1])
