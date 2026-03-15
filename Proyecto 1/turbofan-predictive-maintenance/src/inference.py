# src/inference.py
"""
Inference Engine for RUL Prediction

This module provides the RULInference class to generate Remaining Useful Life
(RUL) predictions using the trained LSTM model.

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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RULInference:
    """
    Inference engine for Remaining Useful Life (RUL) predictions.

    This class encapsulates the trained LSTM model, scaler, and metadata
    required to generate RUL predictions on engine sensor data.

    Attributes:
        project_root (Path): Project root path.
        device (torch.device): Inference device (CPU or CUDA).
        scaler (StandardScaler): Scaler for feature normalization.
        feature_cols (list): Expected feature column names.
        model (LSTMPredictor): Loaded LSTM model.

    Example:
        >>> from pathlib import Path
        >>> inference_engine = RULInference(Path("."))
        >>> engine_data = df[df['id'] == 42].sort_values('cycle')
        >>> rul = inference_engine.predict(engine_data)
        >>> print(f"Predicted RUL: {rul:.1f} cycles")
        Predicted RUL: 52.3 cycles
    """

    def __init__(self, project_root: Union[str, Path]):
        """
        Initialize the inference engine by loading model artifacts.

        Args:
            project_root (Union[str, Path]): Project root path where
                'models/' and 'data/' directories live.

        Raises:
            FileNotFoundError: If model files are missing.
            RuntimeError: If the model fails to load.
        """
        self.project_root = Path(project_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing RULInference on device: {self.device}")
        logger.info(f"Project root: {self.project_root}")

        try:
            # 1. Load metadata and scaler
            scaler_path = self.project_root / "models" / "scaler_v1.pkl"
            feature_cols_path = self.project_root / "models" / "feature_cols_v1.pkl"
            model_path = self.project_root / "models" / "lstm_model_v1.pth"

            # Validate that required files exist
            for path in [scaler_path, feature_cols_path, model_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Archivo requerido no encontrado: {path}")

            self.scaler = joblib.load(scaler_path)
            self.feature_cols = joblib.load(feature_cols_path)

            logger.info(f"Scaler loaded: {scaler_path.name}")
            logger.info(f"Expected features: {len(self.feature_cols)} columns")

            # 2. Initialize and load the model
            input_dim = len(self.feature_cols)
            self.model = LSTMPredictor(input_dim=input_dim, hidden_dim=64, num_layers=2)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"LSTM model loaded: {model_path.name}")
            logger.info(f"Input dimension: {input_dim}")
            logger.info("Inference engine ready")

        except FileNotFoundError as e:
            logger.error(f"Error loading files: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise RuntimeError(f"Failed to initialize RULInference: {e}")

    def predict(
        self, engine_data: pd.DataFrame, sequence_length: int = 30
    ) -> Optional[float]:
        """
        Predict RUL for an engine given its historical data.

        Args:
            engine_data (pd.DataFrame): Engine data with at least 'sequence_length'
                rows and all columns in 'feature_cols'.
            sequence_length (int, optional): Number of consecutive cycles to use
                for prediction. Default: 30.

        Returns:
            Optional[float]: Predicted RUL in cycles (>= 0), or None if there is
                not enough data.

        Raises:
            ValueError: If required columns are missing in engine_data.
            RuntimeError: If inference fails.

        Notes:
            - DataFrame must be sorted by cycle.
            - If len(engine_data) < sequence_length, returns None.
            - Negative predictions are clipped to 0.

        Example:
            >>> engine_data = df[df['id'] == 1].sort_values('cycle')
            >>> rul = inference_engine.predict(engine_data, sequence_length=30)
            >>> if rul is not None:
            ...     print(f"RUL: {rul:.1f} cycles")
        """
        try:
            # Validation: enough data
            if len(engine_data) < sequence_length:
                logger.warning(
                    f"Insufficient data for prediction: {len(engine_data)} < {sequence_length}"
                )
                return None

            # Validation: required columns
            missing_cols = set(self.feature_cols) - set(engine_data.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in engine_data: {missing_cols}")

            # Select and scale the correct columns
            data_to_scale = engine_data[self.feature_cols].tail(sequence_length)

            # Check for NaN values
            if data_to_scale.isnull().any().any():
                logger.error("Data contains NaN values")
                raise ValueError("engine_data contains NaN values in required features")

            scaled_data = self.scaler.transform(data_to_scale)

            # Convert to tensor (1, seq_len, num_features)
            input_tensor = (
                torch.tensor(scaled_data, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            # Run prediction
            with torch.no_grad():
                prediction = self.model(input_tensor).cpu().item()

            # Ensure RUL is not negative
            prediction = max(0.0, prediction)

            logger.debug(f"Prediction completed: {prediction:.2f} cycles")

            return prediction

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_batch(
        self, engine_ids: list, data_df: pd.DataFrame, sequence_length: int = 30
    ) -> dict:
        """
        Predict RUL for multiple engines.

        Args:
            engine_ids (list): List of engine IDs.
            data_df (pd.DataFrame): DataFrame with data for all engines.
            sequence_length (int, optional): Sequence length. Default: 30.

        Returns:
            dict: Dictionary {engine_id: rul_prediction}.
                  Values are None for engines without enough data.

        Example:
            >>> results = inference_engine.predict_batch([1, 2, 3], df)
            >>> print(results)
            {1: 45.3, 2: 78.2, 3: 12.5}
        """
        results = {}
        logger.info(f"Batch prediction for {len(engine_ids)} engines")

        for engine_id in engine_ids:
            engine_df = data_df[data_df["id"] == engine_id].sort_values("cycle")
            try:
                results[engine_id] = self.predict(engine_df, sequence_length)
            except Exception as e:
                logger.error(f"Error predicting engine {engine_id}: {e}")
                results[engine_id] = None

        logger.info(f"Batch prediction completed: {len(results)} results")
        return results
