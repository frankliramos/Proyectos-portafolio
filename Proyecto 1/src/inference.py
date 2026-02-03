# src/inference.py
"""
Inference Engine for RUL Prediction

Este módulo proporciona la clase RULInference para realizar predicciones
de Remaining Useful Life (RUL) usando el modelo LSTM entrenado.

Author: Franklin Ramos
Date: 2026-02-03
"""

import torch
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.models import LSTMPredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RULInference:
    """
    Motor de inferencia para predicciones de Remaining Useful Life (RUL).
    
    Esta clase encapsula el modelo LSTM entrenado, el scaler y los metadatos
    necesarios para realizar predicciones de RUL en datos de sensores de motores.
    
    Attributes:
        project_root (Path): Ruta raíz del proyecto.
        device (torch.device): Dispositivo para inferencia (CPU o CUDA).
        scaler (StandardScaler): Scaler para normalizar features.
        feature_cols (list): Nombres de columnas de features esperadas.
        model (LSTMPredictor): Modelo LSTM cargado.
    
    Example:
        >>> from pathlib import Path
        >>> inference_engine = RULInference(Path("."))
        >>> engine_data = df[df['id'] == 42].sort_values('cycle')
        >>> rul = inference_engine.predict(engine_data)
        >>> print(f"RUL predicho: {rul:.1f} ciclos")
        RUL predicho: 52.3 ciclos
    """
    
    def __init__(self, project_root: Union[str, Path]):
        """
        Inicializa el motor de inferencia cargando modelo y artefactos.
        
        Args:
            project_root (Union[str, Path]): Ruta raíz del proyecto donde se
                encuentran los directorios 'models/' y 'data/'.
        
        Raises:
            FileNotFoundError: Si no se encuentran los archivos del modelo.
            RuntimeError: Si hay errores al cargar el modelo.
        """
        self.project_root = Path(project_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Inicializando RULInference en dispositivo: {self.device}")
        logger.info(f"Ruta del proyecto: {self.project_root}")
        
        try:
            # 1. Cargar metadatos y scaler
            scaler_path = self.project_root / "models" / "scaler_v1.pkl"
            feature_cols_path = self.project_root / "models" / "feature_cols_v1.pkl"
            model_path = self.project_root / "models" / "lstm_model_v1.pth"
            
            # Validar que los archivos existen
            for path in [scaler_path, feature_cols_path, model_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Archivo requerido no encontrado: {path}")
            
            self.scaler = joblib.load(scaler_path)
            self.feature_cols = joblib.load(feature_cols_path)
            
            logger.info(f"Scaler cargado: {scaler_path.name}")
            logger.info(f"Features esperadas: {len(self.feature_cols)} columnas")
            
            # 2. Inicializar y cargar el modelo
            input_dim = len(self.feature_cols)
            self.model = LSTMPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Modelo LSTM cargado: {model_path.name}")
            logger.info(f"Dimensión de entrada: {input_dim}")
            logger.info("Motor de inferencia listo")
            
        except FileNotFoundError as e:
            logger.error(f"Error al cargar archivos: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al inicializar: {e}")
            raise RuntimeError(f"No se pudo inicializar RULInference: {e}")

    def predict(
        self, 
        engine_data: pd.DataFrame, 
        sequence_length: int = 30
    ) -> Optional[float]:
        """
        Predice el RUL para un motor dado su historial de datos.
        
        Args:
            engine_data (pd.DataFrame): DataFrame con datos del motor, debe contener
                al menos 'sequence_length' filas y todas las columnas en 'feature_cols'.
            sequence_length (int, optional): Número de ciclos consecutivos a usar
                para la predicción. Default: 30.
        
        Returns:
            Optional[float]: RUL predicho en ciclos (≥ 0), o None si no hay suficientes datos.
        
        Raises:
            ValueError: Si faltan columnas requeridas en engine_data.
            RuntimeError: Si hay errores durante la inferencia.
        
        Notes:
            - El DataFrame debe estar ordenado por ciclo temporal.
            - Si len(engine_data) < sequence_length, retorna None.
            - Las predicciones negativas se ajustan a 0.
        
        Example:
            >>> engine_data = df[df['id'] == 1].sort_values('cycle')
            >>> rul = inference_engine.predict(engine_data, sequence_length=30)
            >>> if rul is not None:
            ...     print(f"RUL: {rul:.1f} ciclos")
        """
        try:
            # Validación: suficientes datos
            if len(engine_data) < sequence_length:
                logger.warning(
                    f"Datos insuficientes para predicción: {len(engine_data)} < {sequence_length}"
                )
                return None
            
            # Validación: columnas requeridas
            missing_cols = set(self.feature_cols) - set(engine_data.columns)
            if missing_cols:
                raise ValueError(
                    f"Columnas faltantes en engine_data: {missing_cols}"
                )
            
            # Seleccionar y escalar las columnas correctas
            data_to_scale = engine_data[self.feature_cols].tail(sequence_length)
            
            # Verificar valores NaN
            if data_to_scale.isnull().any().any():
                logger.error("Datos contienen valores NaN")
                raise ValueError("engine_data contiene valores NaN en features requeridas")
            
            scaled_data = self.scaler.transform(data_to_scale)
            
            # Convertir a tensor (1, seq_len, num_features)
            input_tensor = torch.tensor(
                scaled_data, 
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Realizar predicción
            with torch.no_grad():
                prediction = self.model(input_tensor).cpu().item()
            
            # Asegurar que RUL no sea negativo
            prediction = max(0.0, prediction)
            
            logger.debug(f"Predicción realizada: {prediction:.2f} ciclos")
            
            return prediction
            
        except ValueError as e:
            logger.error(f"Error de validación: {e}")
            raise
        except Exception as e:
            logger.error(f"Error durante inferencia: {e}")
            raise RuntimeError(f"Fallo en la predicción: {e}")
    
    def predict_batch(
        self,
        engine_ids: list,
        data_df: pd.DataFrame,
        sequence_length: int = 30
    ) -> dict:
        """
        Predice RUL para múltiples motores.
        
        Args:
            engine_ids (list): Lista de IDs de motores.
            data_df (pd.DataFrame): DataFrame con datos de todos los motores.
            sequence_length (int, optional): Longitud de secuencia. Default: 30.
        
        Returns:
            dict: Diccionario {engine_id: rul_prediction}. 
                  Valores None para motores sin suficientes datos.
        
        Example:
            >>> results = inference_engine.predict_batch([1, 2, 3], df)
            >>> print(results)
            {1: 45.3, 2: 78.2, 3: 12.5}
        """
        results = {}
        logger.info(f"Predicción batch para {len(engine_ids)} motores")
        
        for engine_id in engine_ids:
            engine_df = data_df[data_df['id'] == engine_id].sort_values('cycle')
            try:
                results[engine_id] = self.predict(engine_df, sequence_length)
            except Exception as e:
                logger.error(f"Error prediciendo motor {engine_id}: {e}")
                results[engine_id] = None
        
        logger.info(f"Predicción batch completada: {len(results)} resultados")
        return results