# src/models.py
"""
Deep Learning Models for RUL Prediction

This module contains neural network architectures for Remaining Useful Life
(RUL) prediction in turbofan engines.

Author: Franklin Ramos
Date: 2026-02-03
"""

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    LSTM model for Remaining Useful Life (RUL) prediction.

    This architecture uses stacked LSTM layers to capture temporal
    dependencies in sensor sequences, followed by a fully connected
    layer for the final prediction.

    Architecture:
        Input -> LSTM(1) -> Dropout -> LSTM(2) -> Dropout -> FC -> Output

    Args:
        input_dim (int): Number of input features (sensors + settings).
        hidden_dim (int, optional): LSTM hidden state size. Default: 64.
        num_layers (int, optional): Number of stacked LSTM layers. Default: 2.
        output_dim (int, optional): Output dimension (1 for RUL). Default: 1.
        dropout (float, optional): Dropout rate between LSTM layers. Default: 0.2.

    Attributes:
        hidden_dim (int): Hidden state size.
        num_layers (int): Number of LSTM layers.
        lstm (nn.LSTM): Stacked LSTM layers.
        fc (nn.Linear): Fully connected output layer.

    Example:
        >>> model = LSTMPredictor(input_dim=17, hidden_dim=64, num_layers=2)
        >>> x = torch.randn(32, 30, 17)  # (batch, seq_len, features)
        >>> output = model(x)  # (32, 1) - RUL predictions
        >>> print(output.shape)
        torch.Size([32, 1])
    """

    def __init__(
        self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2
    ):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        # batch_first=True means the input is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: RUL predictions of shape (batch_size, output_dim).

        Notes:
            - Uses only the last time step of the LSTM output for prediction.
            - Hidden states (h0, c0) are initialized to zeros.
        """
        # Initialize hidden states (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward pass through the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Use only the last time step to predict RUL
        out = self.fc(out[:, -1, :])
        return out
