# app.py
"""
Streamlit Dashboard for Predictive Maintenance

Dashboard interactivo para monitoreo de salud de motores turbofan
y predicción de Remaining Useful Life (RUL).

Author: Franklin Ramos
Date: 2026-02-03
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.inference import RULInference

# ---------------------------
# Configuración de logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Página & estilo
# ---------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide",
    page_icon="✈️",
    initial_sidebar_state="expanded"
)
sns.set_style("darkgrid")

# ---------------------------
# Cargar motor de inferencia
# ---------------------------
PROJECT_ROOT = Path(__file__).parent

@st.cache_resource
def load_inference_engine() -> Optional[RULInference]:
    """Carga el motor de inferencia con manejo de errores."""
    try:
        logger.info("Cargando motor de inferencia...")
        engine = RULInference(PROJECT_ROOT)
        logger.info("Motor de inferencia cargado exitosamente")
        return engine
    except FileNotFoundError as e:
        logger.error(f"Archivos del modelo no encontrados: {e}")
        st.error(f"❌ Error: No se encontraron los archivos del modelo. {e}")
        st.stop()
    except Exception as e:
        logger.error(f"Error al cargar motor de inferencia: {e}")
        st.error(f"❌ Error inesperado al cargar el modelo: {e}")
        st.stop()

infer_engine = load_inference_engine()

# ---------------------------
# Cargar y preparar datos
# ---------------------------
@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Carga y valida datos procesados."""
    try:
        data_path = PROJECT_ROOT / "data" / "processed" / "fd001_test_prepared.parquet"
        if not data_path.exists():
            logger.error(f"Archivo de datos no encontrado: {data_path}")
            st.error(f"❌ Error: Archivo de datos no encontrado en {data_path}")
            return None
        
        logger.info(f"Cargando datos desde {data_path}")
        df = pd.read_parquet(data_path)
        
        # Normalizar nombres (si aún no están normalizados)
        if 'unit_id' in df.columns:
            df = df.rename(columns={'unit_id': 'id'})
        if 'time_cycles' in df.columns:
            df = df.rename(columns={'time_cycles': 'cycle'})
        
        # Alineamos columnas para evitar errores con el scaler/modelo
        df = df.reset_index(drop=True)
        
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
        
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        st.error(f"❌ Error al cargar datos: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# Validaciones mínimas
if 'id' not in df.columns or 'cycle' not in df.columns:
    logger.error("DataFrame no contiene columnas requeridas 'id' y 'cycle'")
    st.error("❌ El DataFrame cargado no contiene las columnas 'id' y 'cycle'. Revisa el archivo procesado.")
    st.stop()

engine_ids = np.sort(df['id'].unique())
logger.info(f"Dataset listo: {len(engine_ids)} motores únicos")

# ---------------------------
# SIDEBAR: opciones globales
# ---------------------------
st.sidebar.header("⚙️ Configuración")

# Información del modelo
with st.sidebar.expander("ℹ️ Información del Modelo", expanded=False):
    st.markdown("""
    **Modelo:** LSTM v1.0  
    **Arquitectura:** 2 capas, 64 unidades ocultas  
    **Dataset:** NASA CMAPSS FD001  
    **Métricas:**
    - MAE: 14.2 ciclos
    - RMSE: 19.7 ciclos
    - R²: 0.78
    
    **Última actualización:** Febrero 2026
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Selección de Motor")
selected_id = st.sidebar.selectbox("ID del Motor", engine_ids, help="Seleccione el motor a monitorear")

# Umbrales para clasificación del estado (personalizables)
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Umbrales de Estado")
critical_thr = st.sidebar.slider(
    "🔴 Crítico (RUL <)", 
    min_value=0, 
    max_value=200, 
    value=30, 
    step=5,
    help="Motores con RUL menor a este valor se marcan como críticos"
)
warning_thr = st.sidebar.slider(
    "🟡 Precaución (RUL <)", 
    min_value=0, 
    max_value=300, 
    value=70, 
    step=5,
    help="Motores con RUL menor a este valor se marcan con precaución"
)

# Visualización de ciclos
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Rango de Visualización")
show_full_cycles = st.sidebar.checkbox("Mostrar todos los ciclos", value=True)
min_cycle, max_cycle = int(df['cycle'].min()), int(df['cycle'].max())
if show_full_cycles:
    cycle_range = (min_cycle, max_cycle)
else:
    # allow user to pick subrange
    cycle_range = st.sidebar.select_slider(
        "Rango de ciclos",
        options=list(range(min_cycle, max_cycle + 1)),
        value=(max(min_cycle, max_cycle - 100), max_cycle)
    )

# Sensores por defecto en el multiselect
available_sensors = [c for c in df.columns if c.startswith("sensor")]
default_sensors = [s for s in ['sensor_4', 'sensor_11', 'sensor_12'] if s in available_sensors]

# Opción de recálculo de todas las predicciones (cached)
st.sidebar.markdown("---")
recompute_all = st.sidebar.button("🔄 Recalcular RUL", help="Recalcular predicciones para todos los motores")

# ---------------------------
# Helper: calcular predicciones para todos los engines (cached)
# ---------------------------
@st.cache_data
def compute_all_predictions(ids: np.ndarray) -> Dict[int, Optional[float]]:
    """Calcula predicciones RUL para todos los motores con manejo de errores."""
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, mid in enumerate(ids):
        try:
            status_text.text(f"Procesando motor {mid} ({idx+1}/{len(ids)})...")
            engine_df = df[df['id'] == mid].sort_values('cycle')
            pred = infer_engine.predict(engine_df)
            results[int(mid)] = float(pred) if pred is not None else np.nan
        except Exception as e:
            logger.error(f"Error prediciendo motor {mid}: {e}")
            results[int(mid)] = np.nan
        
        progress_bar.progress((idx + 1) / len(ids))
    
    progress_bar.empty()
    status_text.empty()
    logger.info(f"Predicciones completadas: {sum(~np.isnan(list(results.values())))} exitosas de {len(ids)}")
    return results

# If user asks to recompute, clear cache then recompute
if recompute_all:
    compute_all_predictions.clear()
    logger.info("Cache de predicciones limpiado")

with st.spinner("⏳ Calculando RUL predicho para todos los motores..."):
    all_preds = compute_all_predictions(engine_ids)

# ---------------------------
# Datos del motor seleccionado
# ---------------------------
engine_data = df[df['id'] == selected_id].sort_values('cycle').reset_index(drop=True)

# Validar que hay datos
if engine_data.empty:
    st.error(f"❌ No se encontraron datos para el motor {selected_id}")
    st.stop()

# Aplicar rango de ciclos seleccionado
engine_data_filtered = engine_data[
    (engine_data['cycle'] >= cycle_range[0]) & 
    (engine_data['cycle'] <= cycle_range[1])
]

if engine_data_filtered.empty:
    st.warning(f"⚠️ No hay datos para el motor {selected_id} en el rango de ciclos seleccionado.")
    st.stop()

current_cycle = int(engine_data['cycle'].max())
logger.info(f"Motor seleccionado: {selected_id}, ciclos totales: {len(engine_data)}, ciclo actual: {current_cycle}")

# Predicción para motor seleccionado (usar datos completos del motor, no filtrados)
try:
    prediction = all_preds.get(selected_id, np.nan)
    prediction_val = float(prediction) if prediction is not None else np.nan
except Exception as e:
    logger.error(f"Error obteniendo predicción para motor {selected_id}: {e}")
    prediction_val = np.nan

# Obtener RUL real si está presente (último ciclo)
real_rul = None
if 'RUL' in engine_data.columns:
    # Si RUL existe en datos procesados, tomar último (puede estar clippeado)
    real_rul = float(engine_data['RUL'].values[-1])
    logger.debug(f"RUL real disponible para motor {selected_id}: {real_rul:.1f}")

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
st.markdown("**Predicción de Vida Útil Remanente (RUL) — LSTM Neural Network**")
st.markdown(f"*Motor seleccionado: ID {selected_id}* | *Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
st.markdown("---")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    st.metric("🔧 Motor ID", selected_id)

with col2:
    st.metric("⏱️ Ciclos Actuales", current_cycle)

with col3:
    if np.isnan(prediction_val):
        st.metric("🎯 RUL Predicho", "N/A")
    else:
        delta = None
        if real_rul is not None:
            delta = f"{(prediction_val - real_rul):.1f}"
        st.metric("🎯 RUL Predicho", f"{prediction_val:.1f} ciclos", delta=delta)
    
    if real_rul is not None:
        st.caption(f"RUL Real: {real_rul:.1f} ciclos")

with col4:
    st.metric("📊 Estado del Activo", state_label)

# ---------------------------
# Distribución de RUL para todos los motores
# ---------------------------
st.markdown("---")
st.subheader("📈 Distribución de RUL Predicho - Flota Completa")

all_pred_series = pd.Series(all_preds).dropna()
if all_pred_series.empty:
    st.warning("⚠️ No hay predicciones disponibles para mostrar.")
else:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(all_pred_series, bins=30, kde=True, ax=ax, color="#2b8cbe", alpha=0.7)
        
        # Agregar líneas de umbral
        ax.axvline(critical_thr, color='red', linestyle='--', linewidth=2, label=f'Crítico (<{critical_thr})')
        ax.axvline(warning_thr, color='orange', linestyle='--', linewidth=2, label=f'Precaución (<{warning_thr})')
        
        ax.set_xlabel("RUL predicho (ciclos)", fontsize=12)
        ax.set_ylabel("Número de motores", fontsize=12)
        ax.set_title("Distribución de RUL en la Flota", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col_right:
        # KPIs resumen
        n_total = len(engine_ids)
        n_nan = int(pd.Series(all_preds).isna().sum())
        n_critical = int((all_pred_series < critical_thr).sum())
        n_warning = int(((all_pred_series >= critical_thr) & (all_pred_series < warning_thr)).sum())
        n_healthy = int((all_pred_series >= warning_thr).sum())
        
        st.metric("🔧 Total Motores", n_total)
        st.metric("⚪ Sin Predicción", n_nan)
        st.metric("🔴 Críticos", n_critical, 
                 delta=f"{(n_critical/n_total*100):.1f}%", 
                 delta_color="inverse")
        st.metric("🟡 Precaución", n_warning,
                 delta=f"{(n_warning/n_total*100):.1f}%",
                 delta_color="off")
        st.metric("🟢 Saludables", n_healthy,
                 delta=f"{(n_healthy/n_total*100):.1f}%",
                 delta_color="normal")

# ---------------------------
# Graficar sensores (toda la ventana o subrange)
# ---------------------------
st.markdown("---")
st.subheader(f"📊 Evolución de Sensores - Motor {selected_id}")

sensors_to_plot = st.multiselect(
    "Seleccione sensores para monitorear",
    options=available_sensors,
    default=default_sensors,
    help="Seleccione múltiples sensores para comparar su evolución temporal"
)

if sensors_to_plot:
    fig, ax = plt.subplots(figsize=(12, 5))
    for sensor in sensors_to_plot:
        if sensor not in engine_data_filtered.columns:
            continue
        ax.plot(
            engine_data_filtered['cycle'], 
            engine_data_filtered[sensor], 
            label=sensor,
            linewidth=2,
            alpha=0.8
        )
    ax.set_xlabel("Ciclos", fontsize=12)
    ax.set_ylabel("Valor Normalizado del Sensor", fontsize=12)
    ax.set_title(f"Evolución Temporal de Sensores - Motor {selected_id}", fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Estadísticas de sensores
    with st.expander("📊 Estadísticas de Sensores Seleccionados"):
        sensor_stats = engine_data_filtered[sensors_to_plot].describe().T
        st.dataframe(sensor_stats, use_container_width=True)
else:
    st.info("ℹ️ Seleccione al menos un sensor para visualizar")

# ---------------------------
# Tabla de datos (todas las filas si lo desea)
# ---------------------------
st.markdown("---")
st.subheader("🔍 Datos del Motor - Vista Detallada")

col_a, col_b = st.columns([2, 1])
with col_a:
    show_rows = st.selectbox(
        "Número de filas a mostrar", 
        options=["Últimas 10", "Últimas 50", "Mostrar todo"], 
        index=0
    )
with col_b:
    export_data = st.checkbox("Habilitar exportación de datos", value=False)

if show_rows == "Últimas 10":
    display_df = engine_data_filtered.tail(10).reset_index(drop=True)
elif show_rows == "Últimas 50":
    display_df = engine_data_filtered.tail(50).reset_index(drop=True)
else:
    display_df = engine_data_filtered.reset_index(drop=True)

st.dataframe(display_df, use_container_width=True, height=300)

if export_data:
    # Opción para descargar datos
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos como CSV",
        data=csv,
        file_name=f'motor_{selected_id}_data_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

# ---------------------------
# Tabla resumen por motor (predicciones)
# ---------------------------
st.markdown("---")
st.subheader("📋 Resumen RUL Predicho - Todos los Motores")

pred_df = pd.DataFrame({
    "Motor ID": list(all_preds.keys()),
    "RUL Predicho (ciclos)": list(all_preds.values())
})
pred_df['RUL Predicho (ciclos)'] = pred_df['RUL Predicho (ciclos)'].astype(float)
pred_df['Estado'] = pred_df['RUL Predicho (ciclos)'].apply(
    lambda x: estado_rul(x, critical_thr, warning_thr) if not np.isnan(x) else "⚪ Sin datos"
)
pred_df = pred_df.sort_values("RUL Predicho (ciclos)")

# Filtro por estado
filter_col1, filter_col2 = st.columns([1, 3])
with filter_col1:
    filter_state = st.multiselect(
        "Filtrar por estado",
        options=["🔴 Crítico", "🟡 Precaución", "🟢 Saludable", "⚪ Sin datos"],
        default=["🔴 Crítico", "🟡 Precaución", "🟢 Saludable", "⚪ Sin datos"]
    )

# Aplicar filtro
filtered_pred_df = pred_df[pred_df['Estado'].isin(filter_state)]

with filter_col2:
    st.info(f"📊 Mostrando {len(filtered_pred_df)} de {len(pred_df)} motores")

# Mostrar la tabla completa con paginación
st.dataframe(
    filtered_pred_df.reset_index(drop=True), 
    use_container_width=True,
    height=400
)

# Botón de exportación
export_predictions = st.checkbox("Exportar predicciones completas", value=False)
if export_predictions:
    csv_preds = filtered_pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar predicciones como CSV",
        data=csv_preds,
        file_name=f'rul_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

# ---------------------------
# Recomendaciones y notas profesionales
# ---------------------------
st.markdown("---")
st.header("💡 Recomendaciones Profesionales")

tab1, tab2, tab3 = st.tabs(["📊 Interpretación", "🔧 Mejores Prácticas", "⚠️ Limitaciones"])

with tab1:
    st.markdown("""
    ### Interpretación de Resultados
    
    **Predicciones RUL:**
    - El RUL mostrado representa los ciclos operacionales estimados antes de falla
    - Basado en modelo LSTM entrenado con datos históricos de degradación
    - **Error esperado:** ±14.2 ciclos (MAE), ±19.7 ciclos (RMSE)
    
    **Estados de Salud:**
    - 🟢 **Saludable (RUL > 70)**: Operación normal, sin acciones requeridas
    - 🟡 **Precaución (30 ≤ RUL ≤ 70)**: Planificar mantenimiento preventivo
    - 🔴 **Crítico (RUL < 30)**: Acción inmediata requerida, alto riesgo de falla
    
    **Consideraciones:**
    - Las predicciones son más precisas cuando RUL < 50 ciclos
    - Valores altos de RUL pueden estar subestimados (efecto de clipping en 125 ciclos)
    """)

with tab2:
    st.markdown("""
    ### Mejores Prácticas Operacionales
    
    **1. Monitoreo Continuo:**
    - Revisar dashboard diariamente para motores en estado crítico
    - Establecer alertas automáticas para cambios de estado
    - Documentar historial de predicciones para análisis de tendencias
    
    **2. Planificación de Mantenimiento:**
    - Para motores críticos: Inspección inmediata + plan de reemplazo
    - Para motores en precaución: Programar mantenimiento en ventana disponible
    - Considerar impacto operacional y disponibilidad de repuestos
    
    **3. Validación de Predicciones:**
    - Comparar predicciones con inspecciones físicas cuando sea posible
    - Registrar falsos positivos/negativos para mejorar modelo
    - Recalibrar umbrales según contexto operacional específico
    
    **4. Integración con Sistemas:**
    - Integrar con CMMS (Computerized Maintenance Management System)
    - Automatizar generación de órdenes de trabajo
    - Crear reportes periódicos para gerencia
    """)

with tab3:
    st.markdown("""
    ### Limitaciones del Modelo
    
    ⚠️ **Importante - Este modelo es para fines educativos y NO debe usarse para:**
    - Decisiones de seguridad de vuelo en tiempo real
    - Certificación regulatoria (FAA, EASA)
    - Operaciones críticas sin validación adicional
    
    **Limitaciones Técnicas:**
    
    1. **Secuencia Mínima**: Requiere 30 ciclos históricos consecutivos
       - Motores nuevos pueden no tener predicciones
       - Considerar modelo alternativo para datos limitados
    
    2. **Condiciones Operacionales**: 
       - Entrenado solo con datos a nivel del mar (FD001)
       - Puede no generalizar a otras condiciones (altitud, clima)
    
    3. **Calidad de Sensores**:
       - Asume sensores funcionales sin deriva
       - No detecta automáticamente sensores defectuosos
       - Validar calidad de datos antes de confiar en predicciones
    
    4. **Incertidumbre**:
       - Predicciones puntuales sin intervalos de confianza
       - No cuantifica incertidumbre del modelo
       - Versión futura incluirá estimación de incertidumbre
    
    5. **Degradación Rápida**:
       - Ventana de 30 ciclos puede suavizar eventos súbitos
       - Complementar con detección de anomalías
    
    **Recomendaciones para Producción:**
    - Validación exhaustiva con datos reales de campo
    - Sistema de monitoreo de drift del modelo
    - Re-entrenamiento periódico con datos actualizados
    - Validación por expertos en mantenimiento
    - Sistema redundante de predicción
    """)

# ---------------------------
# Footer: versión y ayuda rápida
# ---------------------------
st.markdown("---")
st.caption(
    f"**Dashboard v1.0** | Proyecto de Portafolio — Predictive Maintenance (NASA CMAPSS FD001) | "
    f"© 2026 Franklin Ramos | [Ver Documentación](README.md) | [Model Card](MODEL_CARD.md)"
)
st.caption("⚠️ **Disclaimer**: Este proyecto es para fines educativos y de portafolio únicamente.")