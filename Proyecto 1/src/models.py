# src/models.py
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    """
    Modelo LSTM para predicción de Remaining Useful Life (RUL).
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
            dropout=dropout
        )
        
        # Capa totalmente conectada para la salida
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Inicializar estados ocultos (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass a través de la LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Tomamos solo el último paso de tiempo de la secuencia para predecir el RUL
        out = self.fc(out[:, -1, :])
        return out