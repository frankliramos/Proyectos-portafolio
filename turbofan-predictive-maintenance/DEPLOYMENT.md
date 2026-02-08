# 🚀 Deployment Guide - Turbofan Predictive Maintenance Dashboard

[🇪🇸 Versión en Español](#guía-de-despliegue-español)

---

## English Version

### Overview

This guide provides step-by-step instructions for deploying and running the Turbofan Predictive Maintenance Dashboard. The dashboard is a Streamlit application that provides real-time monitoring of aircraft engine health with RUL (Remaining Useful Life) predictions.

### Prerequisites

- **Python**: 3.12 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: ~500MB for project + dependencies
- **Operating System**: Linux, macOS, or Windows

### Quick Start (5 minutes)

#### 1. Clone the Repository

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd Proyectos-portafolio/turbofan-predictive-maintenance
```

#### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (dashboard framework)
- PyTorch (deep learning)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualization)
- scikit-learn (preprocessing)

#### 4. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

### Dashboard Features

Once running, you can:

1. **Monitor Individual Engines**: Select any engine ID to view its health status
2. **View Fleet Overview**: See RUL distribution across all 100 engines
3. **Analyze Sensor Data**: Interactive plots of 21 sensor readings over time
4. **Adjust Thresholds**: Customize critical/warning RUL thresholds
5. **Export Data**: Download predictions and sensor data as CSV

### Configuration Options

#### Adjust Health Thresholds

Use the sidebar to modify:
- **Critical Threshold** (default: < 30 cycles): Red alert status
- **Warning Threshold** (default: < 70 cycles): Yellow caution status
- **Healthy** (default: ≥ 70 cycles): Green normal status

#### Select Sensor Visualization

Choose which sensors to monitor from the multiselect dropdown:
- Default sensors: `sensor_4`, `sensor_11`, `sensor_12`
- Available: All 21 sensors from the NASA CMAPSS dataset

### Troubleshooting

#### Issue: "ModuleNotFoundError"

**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### Issue: "File not found: data/processed/fd001_test_prepared.parquet"

**Solution**: The data files should be included in the repository. If missing, run:
```bash
python prepare_all_data.py
```

#### Issue: "Large model files missing"

**Note**: The large Random Forest models (`rf_baseline_fd001.pkl`, `rf_optimized_fd001.pkl`) were excluded from the repository due to GitHub size limits (254MB total). The dashboard uses the LSTM model which is included. These baseline models were used for comparison only and are not required for the dashboard to function.

#### Issue: Dashboard is slow

**Solution**: 
- Close other applications to free RAM
- Reduce the number of engines displayed
- Use the cycle range filter to limit data points

### Docker Deployment (Alternative)

For containerized deployment:

```bash
# Build the image
docker-compose build

# Run the container
docker-compose up
```

Access at `http://localhost:8501`

### Production Deployment

For production environments, consider:

1. **Hosting Options**:
   - Streamlit Cloud (easiest, free tier available)
   - AWS EC2 / Google Cloud Compute Engine
   - Heroku / Railway
   - Azure App Service

2. **Security**:
   - Add authentication (Streamlit supports OAuth)
   - Use HTTPS
   - Set up environment variables for sensitive data

3. **Monitoring**:
   - Enable Streamlit's built-in analytics
   - Add application logging
   - Set up health check endpoints

### Data Management

#### Training Data
- **Location**: `data/raw/train_FD001.txt`
- **Records**: 20,631 cycles from 100 engines
- **Usage**: Model training and development

#### Test Data
- **Location**: `data/raw/test_FD001.txt`
- **Records**: 13,096 cycles from 100 engines
- **Usage**: Dashboard visualization and predictions

#### Processed Data
- **Format**: Parquet (efficient, compressed)
- **Location**: `data/processed/`
- **Contents**: Preprocessed, normalized sensor data with RUL labels

### Model Information

#### LSTM Model
- **File**: `models/lstm_model_v1.pth`
- **Size**: 224 KB
- **Architecture**: 2-layer LSTM with 64 hidden units
- **Input**: 30-timestep sequences of 21 sensors
- **Output**: RUL prediction in cycles

#### Supporting Files
- `scaler_v1.pkl`: StandardScaler for feature normalization
- `feature_cols_v1.pkl`: List of feature column names

### Performance Tips

1. **First Load**: Initial startup takes ~10-15 seconds as models load
2. **Caching**: Streamlit caches predictions - subsequent views are instant
3. **Refresh**: Use the "Recalculate RUL" button in sidebar to force refresh

### Support & Issues

For questions or issues:
1. Check this deployment guide
2. Review the main README.md
3. Check MODEL_CARD.md for model-specific details
4. Create an issue on GitHub (for bugs/feature requests)

---

## Guía de Despliegue (Español)

### Resumen

Esta guía proporciona instrucciones paso a paso para desplegar y ejecutar el Dashboard de Mantenimiento Predictivo de Turbofan. El dashboard es una aplicación Streamlit que proporciona monitoreo en tiempo real de la salud de motores de aeronaves con predicciones de RUL (Vida Útil Remanente).

### Requisitos Previos

- **Python**: 3.12 o superior
- **RAM**: Mínimo 4GB (8GB recomendado)
- **Almacenamiento**: ~500MB para proyecto + dependencias
- **Sistema Operativo**: Linux, macOS o Windows

### Inicio Rápido (5 minutos)

#### 1. Clonar el Repositorio

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd Proyectos-portafolio/turbofan-predictive-maintenance
```

#### 2. Crear Entorno Virtual

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- Streamlit (framework del dashboard)
- PyTorch (deep learning)
- Pandas, NumPy (procesamiento de datos)
- Matplotlib, Seaborn (visualización)
- scikit-learn (preprocesamiento)

#### 4. Lanzar el Dashboard

```bash
streamlit run app.py
```

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Características del Dashboard

Una vez en ejecución, puedes:

1. **Monitorear Motores Individuales**: Selecciona cualquier ID de motor para ver su estado de salud
2. **Ver Resumen de Flota**: Observa la distribución de RUL en los 100 motores
3. **Analizar Datos de Sensores**: Gráficos interactivos de 21 lecturas de sensores a lo largo del tiempo
4. **Ajustar Umbrales**: Personaliza umbrales de RUL crítico/precaución
5. **Exportar Datos**: Descarga predicciones y datos de sensores como CSV

### Opciones de Configuración

#### Ajustar Umbrales de Salud

Usa la barra lateral para modificar:
- **Umbral Crítico** (predeterminado: < 30 ciclos): Estado de alerta roja
- **Umbral de Precaución** (predeterminado: < 70 ciclos): Estado de precaución amarilla
- **Saludable** (predeterminado: ≥ 70 ciclos): Estado normal verde

#### Seleccionar Visualización de Sensores

Elige qué sensores monitorear desde el menú desplegable:
- Sensores predeterminados: `sensor_4`, `sensor_11`, `sensor_12`
- Disponibles: Los 21 sensores del dataset NASA CMAPSS

### Solución de Problemas

#### Problema: "ModuleNotFoundError"

**Solución**: Asegúrate de que el entorno virtual esté activado y las dependencias instaladas:
```bash
source venv/bin/activate  # o venv\Scripts\activate en Windows
pip install -r requirements.txt
```

#### Problema: "File not found: data/processed/fd001_test_prepared.parquet"

**Solución**: Los archivos de datos deberían estar incluidos en el repositorio. Si faltan, ejecuta:
```bash
python prepare_all_data.py
```

#### Problema: "Large model files missing"

**Nota**: Los modelos grandes de Random Forest (`rf_baseline_fd001.pkl`, `rf_optimized_fd001.pkl`) fueron excluidos del repositorio debido a los límites de tamaño de GitHub (254MB total). El dashboard usa el modelo LSTM que está incluido. Estos modelos baseline se usaron solo para comparación y no son requeridos para que el dashboard funcione.

#### Problema: Dashboard lento

**Solución**: 
- Cierra otras aplicaciones para liberar RAM
- Reduce el número de motores mostrados
- Usa el filtro de rango de ciclos para limitar puntos de datos

### Despliegue con Docker (Alternativa)

Para despliegue containerizado:

```bash
# Construir la imagen
docker-compose build

# Ejecutar el contenedor
docker-compose up
```

Acceso en `http://localhost:8501`

### Información del Modelo

#### Modelo LSTM
- **Archivo**: `models/lstm_model_v1.pth`
- **Tamaño**: 224 KB
- **Arquitectura**: LSTM de 2 capas con 64 unidades ocultas
- **Entrada**: Secuencias de 30 pasos temporales de 21 sensores
- **Salida**: Predicción de RUL en ciclos

#### Archivos de Soporte
- `scaler_v1.pkl`: StandardScaler para normalización de características
- `feature_cols_v1.pkl`: Lista de nombres de columnas de características

### Consejos de Rendimiento

1. **Primera Carga**: El inicio inicial toma ~10-15 segundos mientras se cargan los modelos
2. **Caché**: Streamlit cachea predicciones - vistas subsecuentes son instantáneas
3. **Refrescar**: Usa el botón "Recalcular RUL" en la barra lateral para forzar actualización

### Soporte y Problemas

Para preguntas o problemas:
1. Consulta esta guía de despliegue
2. Revisa el README.md principal
3. Consulta MODEL_CARD.md para detalles específicos del modelo
4. Crea un issue en GitHub (para bugs/solicitudes de características)

---

**Last Updated**: February 2026  
**Version**: 1.0
