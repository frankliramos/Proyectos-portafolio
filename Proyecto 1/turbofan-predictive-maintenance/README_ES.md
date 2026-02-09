# 🔧 Mantenimiento Predictivo: Predicción de RUL de Motores Turbofan

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

[🇬🇧 English Version](./README.md)

## 📋 Descripción del Proyecto

Este proyecto implementa una **solución de mantenimiento predictivo** para motores turbofan de aeronaves utilizando técnicas de deep learning. El objetivo es predecir la **Vida Útil Remanente (RUL)** de los motores basándose en datos de sensores, permitiendo mantenimiento proactivo y previniendo fallas catastróficas.

### 🎯 Problema de Negocio

Las fallas en motores de aeronaves pueden resultar en:
- Riesgos de seguridad para pasajeros y tripulación
- Mantenimiento no programado costoso
- Interrupciones operacionales y retrasos de vuelos
- Pérdida de ingresos debido a tiempo de inactividad de aeronaves

**Solución**: Predecir cuándo fallará un motor antes de que suceda, permitiendo optimizar la programación del mantenimiento.

### 🔬 Enfoque Técnico

- **Modelo**: Red neuronal LSTM (Long Short-Term Memory)
- **Entrada**: Secuencias de 30 pasos temporales de 21 lecturas de sensores
- **Salida**: Vida Útil Remanente (RUL) en ciclos
- **Dataset**: NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) FD001

## 📊 Dataset

### NASA CMAPSS FD001 Dataset

El dataset simula degradación de motores turbofan bajo varias condiciones operacionales:

- **Conjunto de Entrenamiento**: 100 motores con trayectorias completas hasta falla
- **Conjunto de Prueba**: 100 motores con trayectorias parciales (datos censurados)
- **Mediciones de Sensores**: 21 lecturas de sensores por ciclo de tiempo
- **Configuraciones Operacionales**: 3 configuraciones operacionales por medición

**Características Clave**:
- `unit_id`: Identificador único del motor
- `time_cycles`: Paso de tiempo (número de ciclo)
- `op_1`, `op_2`, `op_3`: Configuraciones operacionales
- `s_1` a `s_21`: 21 mediciones de sensores (temperatura, presión, velocidad, etc.)
- `RUL`: Vida Útil Remanente (variable objetivo)

**Fuente de Datos**: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

## 🏗️ Estructura del Proyecto

```
turbofan-predictive-maintenance/
├── app.py                          # Aplicación dashboard Streamlit
├── README.md                       # Documentación en inglés
├── README_ES.md                    # Este archivo (español)
├── requirements.txt                # Dependencias de Python
├── data/
│   ├── raw/                        # Archivos de datos originales NASA CMAPSS
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   └── processed/                  # Datos preprocesados
│       └── fd001_prepared.parquet
├── models/                         # Modelos entrenados y artefactos
│   ├── lstm_model_v1.pth          # Modelo LSTM PyTorch
│   ├── scaler_v1.pkl              # StandardScaler para normalización
│   └── feature_cols_v1.pkl        # Nombres de columnas de características
├── notebooks/                      # Jupyter notebooks para análisis
│   ├── 01_eda_fd001.ipynb         # Análisis Exploratorio de Datos
│   ├── 02_model_baseline_fd001.ipynb  # Modelos baseline (Random Forest)
│   └── 03_model_lstm_fd001.ipynb  # Entrenamiento del modelo LSTM
├── results/                        # Resultados de evaluación del modelo
│   ├── metrics_rf_baseline.csv
│   └── feature_importance_rf.csv
└── src/                            # Módulos de código fuente
    ├── __init__.py
    ├── config.py                   # Configuración y rutas
    ├── data_loading.py             # Utilidades de carga de datos
    ├── features.py                 # Funciones de ingeniería de características
    ├── models.py                   # Arquitecturas de modelos PyTorch
    └── inference.py                # Motor de inferencia para predicciones
```

## 🚀 Primeros Pasos

### Prerequisitos

- Python 3.12 o superior
- Gestor de paquetes pip o conda

### Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd Proyectos-portafolio/turbofan-predictive-maintenance
```

2. **Crear un entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Verificar archivos de datos**: Asegurarse de que el directorio `data/raw/` contiene los archivos NASA CMAPSS.

### Ejecutar el Dashboard

Iniciar el dashboard interactivo de Streamlit:

```bash
streamlit run app.py
```

El dashboard se abrirá en tu navegador en `http://localhost:8501`.

## 📱 Dashboard Interactivo

### 🌐 Visualización del Dashboard

El proyecto incluye un **dashboard interactivo de Streamlit** para monitoreo de salud de motores en tiempo real y predicciones de RUL.

**Acceso Rápido**:
```bash
# Desde el directorio turbofan-predictive-maintenance
streamlit run app.py
```

El dashboard se abre automáticamente en `http://localhost:8501` y proporciona:
- Predicciones de RUL para motores individuales
- Análisis de salud de toda la flota
- Visualización de datos de sensores
- Filtrado y exploración interactiva

![Dashboard de Mantenimiento Predictivo](../../assets/proyecto1-dashboard.png)

### Características del Dashboard

### 1. **Monitoreo de Salud del Motor**
- Predicciones de RUL en tiempo real para motores individuales
- Clasificación del estado de salud (🟢 Saludable | 🟡 Precaución | 🔴 Crítico)
- Conteo de ciclos actual y ciclos remanentes predichos

### 2. **Análisis de Flota Completa**
- Distribución de predicciones de RUL en todos los motores
- Estadísticas resumidas (conteos críticos, precaución, saludables)
- Umbrales de salud ajustables

### 3. **Monitoreo de Sensores**
- Visualización interactiva de datos de sensores
- Comparación de múltiples sensores
- Selección de rango de ciclos personalizable

### 4. **Exploración de Datos**
- Visor de tabla de datos sin procesar
- Tabla resumen de predicciones por motor
- Resultados exportables

### Opciones de Configuración

**Controles de Barra Lateral**:
- Selección de motor
- Ajuste de umbrales de salud (niveles Crítico/Precaución)
- Filtrado de rango de ciclos
- Selección de sensores para visualización

## 🧠 Arquitectura del Modelo

### Red Neuronal LSTM

```python
Arquitectura:
- Capa de Entrada: 21 características × 30 pasos temporales
- Capa LSTM 1: 64 unidades ocultas + dropout (0.2)
- Capa LSTM 2: 64 unidades ocultas + dropout (0.2)
- Capa Densa de Salida: 1 unidad (predicción RUL)
- Función de Pérdida: Error Cuadrático Medio (MSE)
- Optimizador: Adam
```

**¿Por qué LSTM?**
- Captura dependencias temporales en patrones de degradación de sensores
- Maneja secuencias de longitud variable
- Mejor que ML tradicional para predicción de series temporales
- Aprende tendencias de degradación a largo plazo

### Métricas de Rendimiento

| Métrica | Baseline (RF) | Modelo LSTM |
|---------|---------------|-------------|
| **MAE** | ~18.5 ciclos | ~14.2 ciclos |
| **RMSE** | ~24.3 ciclos | ~19.7 ciclos |
| **R²** | 0.68 | 0.78 |

*Nota: Ver `MODEL_CARD.md` para análisis detallado de rendimiento.*

## 🔧 Entrenamiento del Modelo

### Preprocesamiento de Datos

1. **Cálculo de RUL**: Para datos de entrenamiento, RUL = max(ciclos) - ciclo_actual
2. **Recorte de RUL**: Limitado a máximo de 125 ciclos (reduce ruido en datos de vida temprana)
3. **Escalado de Características**: Normalización con StandardScaler
4. **Creación de Secuencias**: Ventanas deslizantes de 30 ciclos consecutivos

### Proceso de Entrenamiento

Ejecutar los notebooks en orden:

1. **EDA**: `notebooks/01_eda_fd001.ipynb`
   - Análisis de correlación de sensores
   - Visualización de patrones de degradación
   - Selección de características

2. **Modelos Baseline**: `notebooks/02_model_baseline_fd001.ipynb`
   - Regresión Random Forest
   - Análisis de importancia de características
   - Ajuste de hiperparámetros

3. **Entrenamiento LSTM**: `notebooks/03_model_lstm_fd001.ipynb`
   - Preparación de secuencias
   - Definición de arquitectura del modelo
   - Entrenamiento con early stopping
   - Evaluación del modelo

## 📈 Ejemplos de Uso

### API de Python

```python
from src.inference import RULInference
from src.data_loading import load_fd001_train
from pathlib import Path

# Inicializar motor de inferencia
project_root = Path(__file__).parent
inference_engine = RULInference(project_root)

# Cargar datos para un motor específico
df = load_fd001_train()
engine_data = df[df['unit_id'] == 42].sort_values('time_cycles')

# Predecir RUL
predicted_rul = inference_engine.predict(engine_data, sequence_length=30)
print(f"RUL Predicho: {predicted_rul:.1f} ciclos")
```

### Predicciones por Lotes

```python
import pandas as pd

# Predecir para todos los motores
results = {}
for engine_id in df['unit_id'].unique():
    engine_df = df[df['unit_id'] == engine_id].sort_values('time_cycles')
    results[engine_id] = inference_engine.predict(engine_df)

# Crear DataFrame resumen
predictions_df = pd.DataFrame.from_dict(results, orient='index', columns=['RUL'])
predictions_df.to_csv('fleet_predictions.csv')
```

## 🔍 Hallazgos Clave

### Análisis de Sensores

**Sensores Más Importantes para Predicción de RUL**:
1. `s_4` - Alta correlación con degradación
2. `s_11` - Medición crítica de temperatura
3. `s_12` - Indicador de presión
4. `s_15` - Métrica de rendimiento
5. `s_7` - Eficiencia operacional

**Sensores de Baja Varianza** (excluidos del modelo):
- `s_1`, `s_5`, `s_6`, `s_10`, `s_16`, `s_18`, `s_19`: Valores constantes o casi constantes

### Patrones de Degradación

- **Vida Temprana** (RUL > 125 ciclos): Los sensores muestran comportamiento estable
- **Vida Media** (50 < RUL < 125): Comienza degradación gradual
- **Fin de Vida** (RUL < 50): Degradación rápida, valores de sensores divergen significativamente

## 🎯 Impacto de Negocio

### Propuesta de Valor

1. **Ahorro de Costos**: Reducir mantenimiento no programado en 30-40%
2. **Seguridad**: Prevenir fallas en vuelo mediante detección temprana
3. **Optimización**: Programar mantenimiento durante tiempo de inactividad planificado
4. **Utilización de Activos**: Extender vida del motor mediante timing de reemplazo óptimo

### Estrategia de Despliegue

**Enfoque Recomendado**:
- Desplegar como microservicio API (FastAPI/Flask)
- Dashboard de monitoreo en tiempo real para equipos de mantenimiento
- Alertas automatizadas cuando motores entren en estado crítico
- Integración con sistemas de gestión de mantenimiento existentes

## 🛠️ Mejoras Futuras

### Corto Plazo
- [ ] Agregar intervalos de confianza a predicciones (Monte Carlo Dropout)
- [ ] Implementar versionado de modelos y pruebas A/B
- [ ] Agregar detección de anomalías para fallas de sensores
- [ ] Crear reportes automatizados (PDF/email)

### Largo Plazo
- [ ] Soporte multi-tipo de motor (FD002, FD003, FD004)
- [ ] Modelado de conjunto (LSTM + Transformer)
- [ ] Transfer learning para nuevos tipos de motores
- [ ] Integración de datos de streaming en tiempo real
- [ ] App móvil para técnicos de campo

## 📚 Referencias

1. **Dataset**: Saxena, A., & Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository.

2. **Artículo**: Zheng, S., et al. (2017). "Long Short-Term Memory Network for Remaining Useful Life estimation". IEEE International Conference on Prognostics and Health Management.

3. **CMAPSS**: Ramasso, E., & Saxena, A. (2014). "Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets". International Journal of Prognostics and Health Management.

## 👤 Autor

**Franklin Ramos**
- Portafolio: [GitHub Portfolio](https://github.com/frankliramos/Proyectos-portafolio)

## 📄 Licencia

Este proyecto es parte de un portafolio de ciencia de datos. Ver archivo `LICENSE` para detalles.

## 🙏 Agradecimientos

- NASA Ames Research Center por proporcionar el dataset CMAPSS

---

**Nota**: Este es un proyecto de portafolio con fines educativos y de demostración. Para despliegue en producción, se requerirían validación, testing y medidas de seguridad adicionales.
