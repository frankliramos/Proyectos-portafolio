# src/config.py
"""
Configuración central del proyecto.

Este módulo centraliza rutas y parámetros básicos de configuración,
de modo que puedan reutilizarse en todos los componentes
(carga de datos, entrenamiento, evaluación, dashboard, etc.).
"""

from pathlib import Path

# Ruta raíz del proyecto (asumimos que este archivo está en PROYECTO 1/src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directorios de datos
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Archivos específicos para el dataset FD001
FD001_TRAIN_FILE = RAW_DATA_DIR / "train_FD001.txt"
FD001_TEST_FILE = RAW_DATA_DIR / "test_FD001.txt"
FD001_RUL_FILE = RAW_DATA_DIR / "RUL_FD001.txt"

# Archivo procesado de ejemplo
FD001_PROCESSED_FILE = PROCESSED_DATA_DIR / "fd001_prepared.parquet"

# Parámetros por defecto para ventanas temporales (para modelos secuenciales)
DEFAULT_SEQUENCE_LENGTH = 30
