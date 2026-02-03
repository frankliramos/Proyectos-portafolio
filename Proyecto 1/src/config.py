# src/config.py
"""
Configuración central del proyecto.

Este módulo centraliza rutas, parámetros y configuraciones básicas,
de modo que puedan reutilizarse en todos los componentes
(carga de datos, entrenamiento, evaluación, dashboard, etc.).

Author: Franklin Ramos
Date: 2026-02-03
"""

from pathlib import Path

# ===========================
# RUTAS DEL PROYECTO
# ===========================

# Ruta raíz del proyecto (asumimos que este archivo está en PROYECTO 1/src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorios de datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Directorio de modelos
MODELS_DIR = PROJECT_ROOT / "models"

# Directorio de resultados
RESULTS_DIR = PROJECT_ROOT / "results"

# Directorio de notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ===========================
# ARCHIVOS DE DATOS FD001
# ===========================

FD001_TRAIN_FILE = RAW_DATA_DIR / "train_FD001.txt"
FD001_TEST_FILE = RAW_DATA_DIR / "test_FD001.txt"
FD001_RUL_FILE = RAW_DATA_DIR / "RUL_FD001.txt"
FD001_PROCESSED_FILE = PROCESSED_DATA_DIR / "fd001_prepared.parquet"

# ===========================
# ARCHIVOS DE MODELO
# ===========================

# Versión actual del modelo
MODEL_VERSION = "v1"

# Rutas de artefactos del modelo
LSTM_MODEL_FILE = MODELS_DIR / f"lstm_model_{MODEL_VERSION}.pth"
SCALER_FILE = MODELS_DIR / f"scaler_{MODEL_VERSION}.pkl"
FEATURE_COLS_FILE = MODELS_DIR / f"feature_cols_{MODEL_VERSION}.pkl"

# ===========================
# HIPERPARÁMETROS DEL MODELO
# ===========================

# Secuencias temporales
DEFAULT_SEQUENCE_LENGTH = 30

# Arquitectura LSTM
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

# Entrenamiento
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Preprocesamiento
MAX_RUL_CLIP = 125  # Valor máximo de RUL para clipping en entrenamiento

# ===========================
# CONFIGURACIÓN DEL DASHBOARD
# ===========================

# Umbrales por defecto de estado de salud
DEFAULT_CRITICAL_THRESHOLD = 30
DEFAULT_WARNING_THRESHOLD = 70

# Sensores por defecto a visualizar
DEFAULT_SENSORS = ['sensor_4', 'sensor_11', 'sensor_12']

# ===========================
# METADATOS DEL PROYECTO
# ===========================

PROJECT_NAME = "Turbofan RUL Prediction"
PROJECT_VERSION = "1.0.0"
AUTHOR = "Franklin Ramos"
DESCRIPTION = "Predictive Maintenance using LSTM for NASA CMAPSS Dataset"

# ===========================
# LOGGING
# ===========================

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
