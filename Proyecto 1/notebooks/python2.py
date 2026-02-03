# %%
import sys
from pathlib import Path

# Ruta de la carpeta del notebook
NOTEBOOK_DIR = Path.cwd()

# Asumimos que la estructura es: ROOT/
#   ├─ src/
#   ├─ data/
#   └─ notebooks/  <-- aquí está este notebook
PROJECT_ROOT = NOTEBOOK_DIR.parent

print("NOTEBOOK_DIR:", NOTEBOOK_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)

# Añadir PROJECT_ROOT al sys.path si no está
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("sys.path[0]:", sys.path[0])

import src

print(src)


import sys
from pathlib import Path

project_root = Path.cwd()
if project_root.name == "notebooks":
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Project root:", project_root)
print(sys.path[:5])

# %%
import importlib
import sys

# Intentar importar y capturar el error real
try:
    import src.data_loading
except Exception as e:
    print("ERROR AL IMPORTAR src.data_loading:")
    print(type(e).__name__, ":", e)
    import traceback

    traceback.print_exc()

# %%
import importlib
import sys

# Intentar importar y capturar el error real
try:
    import src.data_loading
except Exception as e:
    print("ERROR AL IMPORTAR src.data_loading:")
    print(type(e).__name__, ":", e)
    import traceback

    traceback.print_exc()

# %%
try:
    from src.config import (
        RAW_DATA_DIR,
        FD001_TRAIN_FILE,
        FD001_TEST_FILE,
        FD001_RUL_FILE,
    )

    print("✓ Importación exitosa desde config.py")
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("FD001_TRAIN_FILE:", FD001_TRAIN_FILE)
except Exception as e:
    print("ERROR al importar desde config.py:")
    print(e)
    import traceback

    traceback.print_exc()

# %%
# ============================================================================
# NOTEBOOK: 02_model_baseline_fd001.ipynb
# PROYECTO: Predictive Maintenance - NASA CMAPSS FD001
# DESCRIPCIÓN: Modelo Baseline (Random Forest) para predicción de RUL
# AUTOR: [Tu Nombre]
# FECHA: 2026-01-28
# ============================================================================

# %% [markdown]
# # Modelo Baseline: Random Forest para Predicción de RUL
#
# ## Objetivo
# Desarrollar un modelo baseline utilizando Random Forest para predecir el
# Remaining Useful Life (RUL) de motores turbofan del dataset NASA CMAPSS FD001.
#
# ## Metodología
# 1. Carga de datos procesados
# 2. Feature Engineering (lag features, rolling stats, trends)
# 3. Selección de sensores relevantes
# 4. Entrenamiento de Random Forest Regressor
# 5. Evaluación con RMSE y NASA RUL Score
# 6. Análisis de importancia de features

# %% [markdown]
# ## 1. Configuración Inicial

# %%
# Configuración de rutas y sys.path
import sys
from pathlib import Path

# Obtener directorio del notebook y raíz del proyecto
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent

# Agregar raíz del proyecto a sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 Notebook Directory: {NOTEBOOK_DIR}")
print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"✅ sys.path configurado correctamente")

# %%
# Imports estándar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Configuración de visualización
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 2)

print("✅ Librerías estándar importadas")

# %%
# Imports del proyecto
from src import config
from src import data_loading as dl
from src import features as ft

print("✅ Módulos del proyecto importados")
print(f"   - config: {config}")
print(f"   - data_loading: {dl}")
print(f"   - features: {ft}")

# %% [markdown]
# ## 2. Carga de Datos

# %%
# Cargar datos de entrenamiento y prueba
print("📥 Cargando datos FD001...")

train_df = dl.load_fd001_train()
test_df = dl.load_fd001_test()
rul_df = dl.load_fd001_rul()

print(f"✅ Datos cargados:")
print(f"   - Train: {train_df.shape}")
print(f"   - Test: {test_df.shape}")
print(f"   - RUL (ground truth): {rul_df.shape}")

# %%
# Agregar RUL al conjunto de entrenamiento
train_df = dl.add_rul_to_train(train_df)
print(f"✅ RUL agregado al conjunto de entrenamiento")
print(f"   - Columnas: {train_df.columns.tolist()}")
print(
    f"   - RUL stats: min={train_df['RUL'].min()}, max={train_df['RUL'].max()}, mean={train_df['RUL'].mean():.2f}"
)

# %%
# (No recargar train_df aquí, ya contiene la columna 'RUL')
# Si necesitas recargar test_df, hazlo aquí si es necesario:
# test_df = dl.load_fd001_test()

print(f"✅ Datos cargados:")
print(f"   - Train: {train_df.shape}")
print(f"   - Test: {test_df.shape}")


# %% [markdown]
# ## 3. Feature Engineering

# %%
# Aplicar feature engineering al conjunto de entrenamiento
print("🔧 Aplicando feature engineering al conjunto de entrenamiento...")

train_features = ft.prepare_features(
    train_df, [1, 3, 5], [5, 10, 20], [5, 10]
)

print(f"✅ Features creados para entrenamiento:")
print(f"   - Shape: {train_features.shape}")
print(f"   - Columnas totales: {len(train_features.columns)}")
print(f"   - Nuevas features: {len(train_features.columns) - len(train_df.columns)}")

# %%
# Aplicar feature engineering al conjunto de prueba
print("🔧 Aplicando feature engineering al conjunto de prueba...")

test_features = ft.prepare_features(
    test_df, [1, 3, 5], [5, 10, 20], [5, 10]
)

print(f"✅ Features creados para prueba:")
print(f"   - Shape: {test_features.shape}")
print(f"   - Columnas totales: {len(test_features.columns)}")

# %%
# Verificar valores nulos después de feature engineering
print("🔍 Verificando valores nulos...")
print(
    f"Train - Nulos por columna:\n{train_features.isnull().sum()[train_features.isnull().sum() > 0]}"
)
print(
    f"\nTest - Nulos por columna:\n{test_features.isnull().sum()[test_features.isnull().sum() > 0]}"
)

# Eliminar filas con valores nulos (generados por lag/rolling)
train_features = train_features.dropna()
test_features = test_features.dropna()

print(f"\n✅ Datos limpios:")
print(f"   - Train: {train_features.shape}")
print(f"   - Test: {test_features.shape}")

# %% [markdown]
# ## 4. Selección de Features

# %%
# Definir sensores relevantes basados en EDA
# Estos sensores mostraron correlación significativa con RUL
RELEVANT_SENSORS = [
    "s_2",
    "s_3",
    "s_4",
    "s_7",
    "s_8",
    "s_9",
    "s_11",
    "s_12",
    "s_13",
    "s_14",
    "s_15",
    "s_17",
    "s_20",
    "s_21",
]

# Seleccionar features: sensores base + features derivados + operational settings
feature_cols = []

# 1. Sensores relevantes
feature_cols.extend(RELEVANT_SENSORS)

# 2. Operational settings (column names in your data are 'op_1', 'op_2', 'op_3')
feature_cols.extend(["op_1", "op_2", "op_3"])

# 3. Lag features de sensores relevantes
for sensor in RELEVANT_SENSORS:
    for lag in [1, 3, 5]:
        col = f"{sensor}_lag_{lag}"
        if col in train_features.columns:
            feature_cols.append(col)

# 4. Rolling mean features
for sensor in RELEVANT_SENSORS:
    for window in [5, 10, 20]:
        col = f"{sensor}_rolling_mean_{window}"
        if col in train_features.columns:
            feature_cols.append(col)

# 5. Rolling std features
for sensor in RELEVANT_SENSORS:
    for window in [5, 10, 20]:
        col = f"{sensor}_rolling_std_{window}"
        if col in train_features.columns:
            feature_cols.append(col)

# 6. Trend features
for sensor in RELEVANT_SENSORS:
    for window in [5, 10]:
        col = f"{sensor}_trend_{window}"
        if col in train_features.columns:
            feature_cols.append(col)

print(f"✅ Features seleccionados: {len(feature_cols)}")
print(f"   - Sensores base: {len(RELEVANT_SENSORS)}")
print(f"   - Operational settings: 3")
print(f"   - Features derivados: {len(feature_cols) - len(RELEVANT_SENSORS) - 3}")

# %%
# Preparar conjuntos de entrenamiento y prueba
X_train = train_features[feature_cols]
y_train = train_features["RUL"]

# Para test: seleccionar solo el último ciclo de cada unidad y asignar true_RUL
test_last_cycles = test_features.groupby("unit_id").tail(1).copy()
test_last_cycles = test_last_cycles.reset_index(drop=True)
test_last_cycles["true_RUL"] = rul_df["RUL"].values

X_test = test_last_cycles[feature_cols]
y_test = test_last_cycles["true_RUL"]

print(f"✅ Conjuntos preparados:")
print(f"   - X_train: {X_train.shape}")
print(f"   - y_train: {y_train.shape}")
print(f"   - X_test: {X_test.shape}")
print(f"   - y_test: {y_test.shape}")

# %%
# Crear conjunto de validación desde entrenamiento
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"✅ Conjuntos de validación creados:")
print(f"   - X_train_split: {X_train_split.shape}")
print(f"   - X_val: {X_val.shape}")
print(f"   - y_train_split: {y_train_split.shape}")
print(f"   - y_val: {y_val.shape}")

# %% [markdown]
# ## 5. Entrenamiento del Modelo Baseline

# %%
# Entrenar Random Forest Regressor
print("🌲 Entrenando Random Forest Regressor...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

rf_model.fit(X_train_split, y_train_split)

print("✅ Modelo entrenado exitosamente")

# %% [markdown]
# ## 6. Evaluación del Modelo

# %%
# Predicciones en conjunto de validación
y_val_pred = rf_model.predict(X_val)

# Métricas de validación
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("📊 Métricas en Conjunto de Validación:")
print(f"   - RMSE: {val_rmse:.2f} ciclos")
print(f"   - MAE: {val_mae:.2f} ciclos")
print(f"   - R²: {val_r2:.4f}")

# %%
# Predicciones en conjunto de prueba
y_test_pred = rf_model.predict(X_test)

# Métricas de prueba
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("📊 Métricas en Conjunto de Prueba:")
print(f"   - RMSE: {test_rmse:.2f} ciclos")
print(f"   - MAE: {test_mae:.2f} ciclos")
print(f"   - R²: {test_r2:.4f}")


# %%
# Calcular NASA RUL Score
def calculate_nasa_score(y_true, y_pred):
    """
    Calcula el NASA RUL Score (función de scoring asimétrica).

    Penaliza más las predicciones tardías (late predictions) que las tempranas.
    - Si predicción < real: error = exp(-diff/13) - 1
    - Si predicción >= real: error = exp(diff/10) - 1

    Parameters:
    -----------
    y_true : array-like
        Valores reales de RUL
    y_pred : array-like
        Valores predichos de RUL

    Returns:
    --------
    float
        NASA RUL Score (menor es mejor)
    """
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    return np.sum(score)


# Calcular NASA Score
val_nasa_score = calculate_nasa_score(y_val, y_val_pred)
test_nasa_score = calculate_nasa_score(y_test, y_test_pred)

print("🎯 NASA RUL Score:")
print(f"   - Validación: {val_nasa_score:.2f}")
print(f"   - Prueba: {test_nasa_score:.2f}")

# %% [markdown]
# ## 7. Análisis de Importancia de Features

# %%
# Obtener importancia de features
feature_importance = pd.DataFrame(
    {"feature": feature_cols, "importance": rf_model.feature_importances_}
).sort_values("importance", ascending=False)

print("🔝 Top 20 Features más importantes:")
print(feature_importance.head(20))

# %%
# Visualizar top 20 features
fig, ax = plt.subplots(figsize=(10, 8))
top_20 = feature_importance.head(20)
ax.barh(range(len(top_20)), top_20["importance"])
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20["feature"])
ax.set_xlabel("Importancia")
ax.set_title("Top 20 Features más Importantes - Random Forest")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Visualización de Predicciones

# %%
# Scatter plot: Predicciones vs Valores Reales (Validación)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Validación
axes[0].scatter(y_val, y_val_pred, alpha=0.5, s=10)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
axes[0].set_xlabel("RUL Real")
axes[0].set_ylabel("RUL Predicho")
axes[0].set_title(f"Validación - RMSE: {val_rmse:.2f}")
axes[0].grid(True, alpha=0.3)

# Prueba
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=10)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[1].set_xlabel("RUL Real")
axes[1].set_ylabel("RUL Predicho")
axes[1].set_title(f"Prueba - RMSE: {test_rmse:.2f}")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Distribución de errores
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Validación
val_errors = y_val_pred - y_val
axes[0].hist(val_errors, bins=50, edgecolor="black", alpha=0.7)
axes[0].axvline(0, color="r", linestyle="--", linewidth=2)
axes[0].set_xlabel("Error de Predicción (ciclos)")
axes[0].set_ylabel("Frecuencia")
axes[0].set_title(f"Distribución de Errores - Validación\nMAE: {val_mae:.2f}")
axes[0].grid(True, alpha=0.3)

# Prueba
test_errors = y_test_pred - y_test
axes[1].hist(test_errors, bins=50, edgecolor="black", alpha=0.7)
axes[1].axvline(0, color="r", linestyle="--", linewidth=2)
axes[1].set_xlabel("Error de Predicción (ciclos)")
axes[1].set_ylabel("Frecuencia")
axes[1].set_title(f"Distribución de Errores - Prueba\nMAE: {test_mae:.2f}")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Análisis de errores por rango de RUL
test_results = pd.DataFrame(
    {
        "true_RUL": y_test,
        "pred_RUL": y_test_pred,
        "error": y_test_pred - y_test,
        "abs_error": np.abs(y_test_pred - y_test),
    }
)

# Crear bins de RUL
test_results["RUL_bin"] = pd.cut(
    test_results["true_RUL"],
    bins=[0, 50, 100, 150, 200, 300],
    labels=["0-50", "50-100", "100-150", "150-200", "200+"],
)

# Calcular métricas por bin
bin_metrics = (
    test_results.groupby("RUL_bin")
    .agg({"abs_error": ["mean", "std", "count"]})
    .round(2)
)

print("📊 Métricas de Error por Rango de RUL:")
print(bin_metrics)

# %%
# Visualizar error por rango de RUL
fig, ax = plt.subplots(figsize=(10, 6))
test_results.boxplot(column="abs_error", by="RUL_bin", ax=ax)
ax.set_xlabel("Rango de RUL (ciclos)")
ax.set_ylabel("Error Absoluto (ciclos)")
ax.set_title("Distribución de Error Absoluto por Rango de RUL")
plt.suptitle("")  # Remover título automático de pandas
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Resumen de Resultados

# %%
# Resumen final
print("=" * 70)
print("RESUMEN DE RESULTADOS - MODELO BASELINE (RANDOM FOREST)")
print("=" * 70)
print(f"\n📊 MÉTRICAS DE VALIDACIÓN:")
print(f"   - RMSE: {val_rmse:.2f} ciclos")
print(f"   - MAE: {val_mae:.2f} ciclos")
print(f"   - R²: {val_r2:.4f}")
print(f"   - NASA Score: {val_nasa_score:.2f}")

print(f"\n📊 MÉTRICAS DE PRUEBA:")
print(f"   - RMSE: {test_rmse:.2f} ciclos")
print(f"   - MAE: {test_mae:.2f} ciclos")
print(f"   - R²: {test_r2:.4f}")
print(f"   - NASA Score: {test_nasa_score:.2f}")

print(f"\n🔧 CONFIGURACIÓN DEL MODELO:")
print(f"   - Algoritmo: Random Forest Regressor")
print(f"   - N° de árboles: 100")
print(f"   - Max depth: 20")
print(f"   - Features utilizados: {len(feature_cols)}")

print(f"\n🎯 PRÓXIMOS PASOS:")
print(f"   1. Optimización de hiperparámetros (GridSearch/RandomSearch)")
print(f"   2. Modelo Deep Learning (LSTM)")
print(f"   3. Dashboard interactivo con Streamlit")
print("=" * 70)

# %% [markdown]
# ## 10. Guardar Modelo y Resultados

# %%
# Guardar modelo entrenado
import joblib

model_path = PROJECT_ROOT / "models" / "rf_baseline_fd001.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(rf_model, model_path)
print(f"✅ Modelo guardado en: {model_path}")

# %%
# Guardar feature importance
feature_importance_path = PROJECT_ROOT / "results" / "feature_importance_rf.csv"
feature_importance_path.parent.mkdir(parents=True, exist_ok=True)

feature_importance.to_csv(feature_importance_path, index=False)
print(f"✅ Feature importance guardado en: {feature_importance_path}")

# %%
# Guardar métricas
metrics = {
    "model": "Random Forest Baseline",
    "val_rmse": val_rmse,
    "val_mae": val_mae,
    "val_r2": val_r2,
    "val_nasa_score": val_nasa_score,
    "test_rmse": test_rmse,
    "test_mae": test_mae,
    "test_r2": test_r2,
    "test_nasa_score": test_nasa_score,
    "n_features": len(feature_cols),
    "n_estimators": 100,
    "max_depth": 20,
}

metrics_df = pd.DataFrame([metrics])
metrics_path = PROJECT_ROOT / "results" / "metrics_rf_baseline.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"✅ Métricas guardadas en: {metrics_path}")

print("\n🎉 Notebook completado exitosamente!")

# %%



