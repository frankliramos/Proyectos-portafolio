# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
from src.inference import RULInference

# ---------------------------
# Página & estilo
# ---------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
sns.set_style("darkgrid")

# ---------------------------
# Cargar motor de inferencia
# ---------------------------
PROJECT_ROOT = Path(__file__).parent

@st.cache_resource
def load_inference_engine() -> RULInference:
    return RULInference(PROJECT_ROOT)

infer_engine = load_inference_engine()

# ---------------------------
# Cargar y preparar datos
# ---------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "fd001_prepared.parquet")
    # Normalizar nombres (si aún no están normalizados)
    df = df.rename(columns={'unit_id': 'id', 'time_cycles': 'cycle'})
    # Alineamos columnas para evitar errores con el scaler/modelo
    df = df.reset_index(drop=True)
    return df

df = load_data()

# Validaciones mínimas
if 'id' not in df.columns or 'cycle' not in df.columns:
    st.error("El DataFrame cargado no contiene las columnas 'id' y 'cycle'. Revisa el archivo procesado.")
    st.stop()

engine_ids = np.sort(df['id'].unique())

# ---------------------------
# SIDEBAR: opciones globales
# ---------------------------
st.sidebar.header("Configuración")
selected_id = st.sidebar.selectbox("Seleccione el ID del Motor", engine_ids)

# Umbrales para clasificación del estado (personalizables)
st.sidebar.markdown("**Umbrales de estado (ajustables)**")
critical_thr = st.sidebar.slider("Crítico < ", min_value=0, max_value=200, value=30, step=5)
warning_thr = st.sidebar.slider("Precaución < ", min_value=0, max_value=300, value=70, step=5)

# Visualización de ciclos
show_full_cycles = st.sidebar.checkbox("Mostrar todos los ciclos (True = sí)", value=True)
min_cycle, max_cycle = int(df['cycle'].min()), int(df['cycle'].max())
if show_full_cycles:
    cycle_range = (min_cycle, max_cycle)
else:
    # allow user to pick subrange
    cycle_range = st.sidebar.select_slider(
        "Rango de ciclos a mostrar",
        options=list(range(min_cycle, max_cycle + 1)),
        value=(max(min_cycle, max_cycle - 100), max_cycle)
    )

# Sensores por defecto en el multiselect
available_sensors = [c for c in df.columns if c.startswith("sensor")]
default_sensors = [s for s in ['sensor_4', 'sensor_11', 'sensor_12'] if s in available_sensors]

# Opción de recálculo de todas las predicciones (cached)
recompute_all = st.sidebar.button("Recalcular RUL para todos los motores")

# ---------------------------
# Helper: calcular predicciones para todos los engines (cached)
# ---------------------------
@st.cache_data
def compute_all_predictions(ids: np.ndarray) -> Dict[int, Optional[float]]:
    results = {}
    for mid in ids:
        engine_df = df[df['id'] == mid].sort_values('cycle')
        try:
            pred = infer_engine.predict(engine_df)
        except Exception:
            pred = None
        results[int(mid)] = float(pred) if pred is not None else np.nan
    return results

# If user asks to recompute, clear cache then recompute
if recompute_all:
    compute_all_predictions.clear()

with st.spinner("Calculando RUL predicho para todos los motores..."):
    all_preds = compute_all_predictions(engine_ids)

# ---------------------------
# Datos del motor seleccionado
# ---------------------------
engine_data = df[df['id'] == selected_id].sort_values('cycle').reset_index(drop=True)
# Aplicar rango de ciclos seleccionado
engine_data = engine_data[(engine_data['cycle'] >= cycle_range[0]) & (engine_data['cycle'] <= cycle_range[1])]

if engine_data.empty:
    st.warning(f"No hay datos para el motor {selected_id} en el rango de ciclos seleccionado.")
    st.stop()

current_cycle = int(engine_data['cycle'].max())

# Predicción para motor seleccionado
prediction = infer_engine.predict(df[df['id'] == selected_id].sort_values('cycle'))  # use full engine data for prediction
prediction_val = float(prediction) if prediction is not None else np.nan

# Obtener RUL real si está presente (último ciclo)
real_rul = None
if 'RUL' in engine_data.columns:
    # Si RUL existe en datos procesados, tomar último (puede estar clippeado)
    real_rul = float(engine_data['RUL'].values[-1])

# ---------------------------
# Función para estado en base a umbrales ajustables
# ---------------------------
def estado_rul(rul: float, crit_thr: int, warn_thr: int) -> str:
    if np.isnan(rul):
        return "⚪ Sin datos"
    if rul < crit_thr:
        return "🔴 Crítico"
    if rul < warn_thr:
        return "🟡 Precaución"
    return "🟢 Saludable"

state_label = estado_rul(prediction_val, critical_thr, warning_thr)

# ---------------------------
# MAIN: KPIs
# ---------------------------
st.title("✈️ Turbofan Engine Health Monitor")
st.markdown("Predicción de Vida Útil Remanente (RUL) por motor — LSTM")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric("Ciclos Actuales", current_cycle)

with col2:
    if np.isnan(prediction_val):
        st.metric("RUL Predicho", "N/A")
    else:
        st.metric("RUL Predicho", f"{prediction_val:.1f} ciclos")

    if real_rul is not None:
        st.caption("RUL (último ciclo) disponible en los datos procesados")
        st.write(f"RUL Real (último ciclo): {real_rul:.1f} ciclos")

with col3:
    st.metric("Estado del Activo", state_label)

# ---------------------------
# Distribución de RUL para todos los motores
# ---------------------------
st.subheader("Distribución de RUL predicho - Todos los motores")

all_pred_series = pd.Series(all_preds).dropna()
if all_pred_series.empty:
    st.write("No hay predicciones disponibles para mostrar.")
else:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.histplot(all_pred_series, bins=30, kde=True, ax=ax, color="#2b8cbe")
    ax.set_xlabel("RUL predicho (ciclos)")
    ax.set_ylabel("Número de motores")
    st.pyplot(fig)

    # KPIs resumen
    n_total = len(engine_ids)
    n_nan = int(pd.Series(all_preds).isna().sum())
    n_critical = int((all_pred_series < critical_thr).sum())
    n_warning = int(((all_pred_series >= critical_thr) & (all_pred_series < warning_thr)).sum())
    n_healthy = int((all_pred_series >= warning_thr).sum())

    st.write(
        f"Total motores: {n_total} | Sin predicción: {n_nan} | "
        f"Críticos (<{critical_thr}): {n_critical} | Precaución (<{warning_thr}): {n_warning} | Saludables: {n_healthy}"
    )

# ---------------------------
# Graficar sensores (toda la ventana o subrange)
# ---------------------------
st.subheader("📈 Evolución de Sensores")

sensors_to_plot = st.multiselect(
    "Seleccione sensores para monitorear",
    options=available_sensors,
    default=default_sensors
)

if sensors_to_plot:
    fig, ax = plt.subplots(figsize=(12, 4))
    for sensor in sensors_to_plot:
        if sensor not in engine_data.columns:
            continue
        ax.plot(engine_data['cycle'], engine_data[sensor], label=sensor)
    ax.set_xlabel("Ciclos")
    ax.set_ylabel("Valor")
    ax.legend(loc="upper right")
    st.pyplot(fig)

# ---------------------------
# Tabla de datos (todas las filas si lo desea)
# ---------------------------
st.subheader("🔎 Datos crudos del motor")
show_rows = st.selectbox("Número de filas a mostrar", options=["Últimas 10", "Últimas 50", "Mostrar todo"], index=0)
if show_rows == "Últimas 10":
    st.dataframe(engine_data.tail(10).reset_index(drop=True))
elif show_rows == "Últimas 50":
    st.dataframe(engine_data.tail(50).reset_index(drop=True))
else:
    st.dataframe(engine_data.reset_index(drop=True))

# ---------------------------
# Tabla resumen por motor (predicciones)
# ---------------------------
st.subheader("Resumen RUL predicho por motor")
pred_df = pd.DataFrame({
    "id": list(all_preds.keys()),
    "rul_pred": list(all_preds.values())
})
pred_df['rul_pred'] = pred_df['rul_pred'].astype(float)
pred_df['state'] = pred_df['rul_pred'].apply(lambda x: estado_rul(x, critical_thr, warning_thr) if not np.isnan(x) else "⚪ Sin datos")
pred_df = pred_df.sort_values("rul_pred")

# Mostrar la tabla completa con paginación
st.dataframe(pred_df.reset_index(drop=True), use_container_width=True)

# ---------------------------
# Recomendaciones y notas profesionales
# ---------------------------
st.markdown("---")
st.header("Recomendaciones profesionales")
st.markdown(
    """
    - El RUL mostrado es la predicción del modelo LSTM entrenado (con RUL clippeado en entrenamiento).
    - Recomendamos:
      1. Verificar la distribución de RUL real vs predicho (para calibrar umbrales).
      2. Implementar un umbral operativo crítico ajustado por tipo de motor y contexto operacional.
      3. Añadir intervalos de confianza si se usa un ensemble o dropout MC para estimar incertidumbre.
      4. Validar en producción con datos en tiempo real y crear alertas automatizadas (email/Slack) para motores en estado crítico.
    - Si notas que muchas unidades aparecen como críticas, puede indicar:
      - Sesgo del modelo hacia subestimación (requeriría recalibrado).
      - RUL clippeado en entrenamiento (el modelo no ve valores > MAX_RUL).
      - Problema con el escalado/columnas (verificar que las columnas de entrada coincidan con las usadas en entrenamiento).
    """
)

# ---------------------------
# Footer: versión y ayuda rápida
# ---------------------------
st.markdown("---")
st.caption("Dashboard creado para el proyecto de portafolio — Predictive Maintenance (NASA CMAPSS FD001).")