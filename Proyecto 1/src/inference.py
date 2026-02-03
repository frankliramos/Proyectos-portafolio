# src/inference.py
import torch
import numpy as np
import joblib
from pathlib import Path
from src.models import LSTMPredictor

class RULInference:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Cargar metadatos y scaler
        self.scaler = joblib.load(self.project_root / "models" / "scaler_v1.pkl")
        self.feature_cols = joblib.load(self.project_root / "models" / "feature_cols_v1.pkl")
        
        # 2. Inicializar y cargar el modelo
        input_dim = len(self.feature_cols)
        self.model = LSTMPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2)
        self.model.load_state_dict(torch.load(self.project_root / "models" / "lstm_model_v1.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, engine_data, sequence_length=30):
        """
        Recibe un DataFrame de un solo motor, toma los últimos ciclos y predice el RUL.
        """
        if len(engine_data) < sequence_length:
            return None # No hay suficientes datos para una secuencia completa
        
        # Seleccionar y escalar las columnas correctas
        data_to_scale = engine_data[self.feature_cols].tail(sequence_length)
        scaled_data = self.scaler.transform(data_to_scale)
        
        # Convertir a tensor (1, seq_len, num_features)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().item()
        
        return max(0, prediction) # El RUL no puede ser negativo