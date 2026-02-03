# src/models.py
"""
Deep Learning Models for RUL Prediction

Este módulo contiene arquitecturas de redes neuronales para
predicción de Remaining Useful Life (RUL) en motores turbofan.

Author: Franklin Ramos
Date: 2026-02-03
"""

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    Modelo LSTM para predicción de Remaining Useful Life (RUL).
    
    Esta arquitectura utiliza capas LSTM apiladas para capturar dependencias
    temporales en secuencias de datos de sensores, seguidas de una capa
    totalmente conectada para la predicción final.
    
    Arquitectura:
        Input → LSTM(1) → Dropout → LSTM(2) → Dropout → FC → Output
    
    Args:
        input_dim (int): Número de features de entrada (sensores + configuraciones).
        hidden_dim (int, optional): Dimensión de los estados ocultos LSTM. Default: 64.
        num_layers (int, optional): Número de capas LSTM apiladas. Default: 2.
        output_dim (int, optional): Dimensión de salida (1 para RUL). Default: 1.
        dropout (float, optional): Tasa de dropout entre capas LSTM. Default: 0.2.
    
    Attributes:
        hidden_dim (int): Dimensión de estados ocultos.
        num_layers (int): Número de capas LSTM.
        lstm (nn.LSTM): Capas LSTM apiladas.
        fc (nn.Linear): Capa totalmente conectada de salida.
    
    Example:
        >>> model = LSTMPredictor(input_dim=17, hidden_dim=64, num_layers=2)
        >>> x = torch.randn(32, 30, 17)  # (batch, seq_len, features)
        >>> output = model(x)  # (32, 1) - predicciones RUL
        >>> print(output.shape)
        torch.Size([32, 1])
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Capa LSTM
        # batch_first=True significa que la entrada es (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Capa totalmente conectada para la salida
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass del modelo.
        
        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch_size, seq_length, input_dim).
        
        Returns:
            torch.Tensor: Predicciones de RUL de forma (batch_size, output_dim).
        
        Notes:
            - Toma solo el último paso temporal de la salida LSTM para predicción.
            - Los estados ocultos (h0, c0) se inicializan en cero.
        """
        # Inicializar estados ocultos (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass a través de la LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Tomamos solo el último paso de tiempo de la secuencia para predecir el RUL
        out = self.fc(out[:, -1, :])
        return out