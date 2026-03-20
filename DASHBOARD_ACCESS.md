# 🖥️ Dashboard Access Guide

## English

### 🌐 Live Demos — No Installation Required

All four dashboards are deployed online. Click to open instantly:

| # | Project | Live Demo |
|---|---------|-----------|
| 1 | 🔧 Turbofan Engine Health Monitor | [🚀 Open Dashboard](https://proyectos-portafolio-fvuxxicflgewt7jxjtdzd.streamlit.app/) |
| 2 | 📈 Sales Forecasting System | [🚀 Open Dashboard](https://proyectos-portafolio-amcgczthtr4a7s3epewp2q.streamlit.app/) |
| 3 | 💬 Financial Sentiment Analysis | [🚀 Open Dashboard](https://proyectos-portafolio-erzcz3etb9efmhgmw8qxep.streamlit.app/) |
| 4 | 📡 Churn Prediction - Telecom | [🚀 Open Dashboard](https://proyectos-portafolio-batbljpwqey6pemu5s2dr7.streamlit.app/) |

> **For clients and recruiters**: The dashboards above are ready to use — no setup needed.

---

### How to View Project Dashboards Locally

This portfolio includes interactive **Streamlit dashboards** for all four projects. Follow these simple steps to view them on your local machine.

---

### 📋 Prerequisites

- **Python 3.8+** installed on your computer
- **Git** for cloning the repository
- **Terminal/Command Prompt** access

---

### 🚀 Quick Start (5 Minutes)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd Proyectos-portafolio
```

#### Step 2: Choose a Project

##### Option A: Proyecto 1 - Predictive Maintenance Dashboard

```bash
cd "turbofan-predictive-maintenance"
pip install -r requirements.txt
streamlit run app.py
```

**What you'll see**: 
- Engine health monitoring
- Remaining Useful Life (RUL) predictions
- Sensor data visualization
- Fleet-wide analytics

**Dashboard opens at**: `http://localhost:8501`

##### Option B: Proyecto 2 - Sales Forecasting Dashboard

```bash
cd "sales-forecasting"
pip install -r requirements.txt
streamlit run app.py
```

**What you'll see**:
- Sales predictions by store and product
- Inventory recommendations
- Forecast accuracy metrics
- Demand drivers analysis

**Dashboard opens at**: `http://localhost:8501`

**Note**: Proyecto 2 includes sample forecast data (`data_forecast.csv`) for immediate demonstration. The dashboard also generates synthetic data automatically if the file is missing.

#### Proyecto 3: Financial Sentiment Analysis Dashboard

```bash
cd "financial-sentiment-analysis"
pip install -r requirements.txt
streamlit run app.py
```

**What you'll see**:
- Real-time financial text sentiment classification (Positive / Neutral / Negative)
- FinBERT model predictions with confidence scores
- Keyword and phrase analysis
- Batch analysis for multiple news items
- Interactive charts and model interpretability

**Dashboard opens at**: `http://localhost:8501`

**Note**: Proyecto 3 includes sample data for demonstration. The dashboard is fully functional without additional data files.

#### Option D: Proyecto 4 - Churn Prediction Dashboard

```bash
cd "customer-churn-prediction"
pip install -r requirements.txt
streamlit run app.py
```

**What you'll see**:
- Individual customer churn risk scoring
- SHAP-based explanation of top churn drivers
- Customer segmentation by risk level
- Financial impact estimation and retention recommendations
- Fleet-level analytics

**Dashboard opens at**: `http://localhost:8501`

**Note**: Proyecto 4 includes sample data for demonstration. The dashboard is fully functional without additional data files.

---

### 🔧 Troubleshooting

#### Issue: "Command 'streamlit' not found"

**Solution**: Make sure you've installed the requirements:
```bash
pip install -r requirements.txt
```

#### Issue: "Module not found" errors

**Solution**: Install project dependencies:
```bash
# For Proyecto 1
cd "turbofan-predictive-maintenance"
pip install -r requirements.txt

# For Proyecto 2
cd "sales-forecasting"
pip install -r requirements.txt

# For Proyecto 3
cd "financial-sentiment-analysis"
pip install -r requirements.txt
```

#### Issue: Dashboard won't open

**Solution**: Manually open in your browser:
```
http://localhost:8501
```

#### Issue: Port already in use

**Solution**: Stop other Streamlit instances or specify a different port:
```bash
streamlit run app.py --server.port 8502
```

---

### 📱 Dashboard Features

#### Proyecto 1: Predictive Maintenance
- ✅ **100% Functional** - All data and models included
- 🔧 Select individual engines (1-100)
- 📊 View 21 sensor measurements
- ⚡ Real-time RUL predictions
- 🎯 Health status indicators (Healthy/Warning/Critical)

#### Proyecto 2: Sales Forecasting
- ✅ **100% Functional** - Demo data included (auto-generated if missing)
- 🏬 Select store (1-5 in demo, up to 54 with full dataset)
- 📦 Choose product category
- 📈 15-day sales forecast
- 💰 Inventory optimization suggestions

### Proyecto 3: Financial Sentiment Analysis
- ✅ **100% Functional** - Demo data included
- 💬 Classify financial text (Positive / Neutral / Negative)
- 📊 FinBERT confidence scores
- 🔍 Keyword and phrase analysis
- 📈 Batch news analysis

#### Proyecto 4: Churn Prediction - Telecom
- ✅ **100% Functional** - Demo data included
- 👤 Individual customer churn risk scoring
- 📊 SHAP-based churn driver explanation
- 🎯 Risk classification (Low/Medium/High)
- 💡 Retention prioritization and financial impact
- ⚡ Real-time predictions

---

### 💼 For Clients & Recruiters

All four dashboards are **deployed online and production-ready**. Access them instantly via the live demo links at the top of this guide — no installation needed.

They demonstrate:

1. **Real-time Analytics**: Interactive data exploration
2. **Business Insights**: Actionable predictions and recommendations
3. **User Experience**: Clean, professional interfaces
4. **Technical Skills**: Python, ML models, data visualization, web apps

---

### 📧 Support

If you encounter any issues accessing the dashboards:

**Franklin Ramos**
- GitHub: [@frankliramos](https://github.com/frankliramos)
- Repository: [Proyectos-portafolio](https://github.com/frankliramos/Proyectos-portafolio)

---

### 🎥 Video Tutorials (Coming Soon)

- [ ] Proyecto 1 Dashboard Walkthrough
- [ ] Proyecto 2 Dashboard Walkthrough
- [ ] Proyecto 3 Dashboard Walkthrough
- [ ] Installation Guide for Windows
- [ ] Installation Guide for Mac/Linux

---

**Last Updated**: March 2026

---

## Espanol

### 🌐 Demos en Vivo — Sin Instalación

Los cuatro dashboards están desplegados online. Haz clic para abrirlos al instante:

| # | Proyecto | Demo en Vivo |
|---|---------|--------------|
| 1 | 🔧 Monitor de Salud de Motores Turbofán | [🚀 Abrir Dashboard](https://proyectos-portafolio-fvuxxicflgewt7jxjtdzd.streamlit.app/) |
| 2 | 📈 Sistema de Pronóstico de Ventas | [🚀 Abrir Dashboard](https://proyectos-portafolio-amcgczthtr4a7s3epewp2q.streamlit.app/) |
| 3 | 💬 Análisis de Sentimiento Financiero | [🚀 Abrir Dashboard](https://proyectos-portafolio-erzcz3etb9efmhgmw8qxep.streamlit.app/) |
| 4 | 📡 Predicción de Churn - Telecomunicaciones | [🚀 Abrir Dashboard](https://proyectos-portafolio-batbljpwqey6pemu5s2dr7.streamlit.app/) |

> **Para clientes y reclutadores**: Los dashboards de arriba están listos para usar — sin configuración necesaria.

---

### Como ver los dashboards de los proyectos

Este portafolio incluye dashboards interactivos de Streamlit. Sigue estos pasos para verlos en tu equipo.

---

### 📋 Requisitos

- Python 3.8+ instalado
- Git para clonar el repositorio
- Terminal disponible

---

### 🚀 Inicio rapido (5 minutos)

#### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd Proyectos-portafolio
```

#### Paso 2: Elegir un proyecto

##### Opcion A: Proyecto 1 - Dashboard de mantenimiento predictivo

```bash
cd "turbofan-predictive-maintenance"
pip install -r requirements.txt
streamlit run app.py
```

**Que veras**:
- Monitoreo de salud del motor
- Predicciones de RUL
- Visualizacion de sensores
- Analitica de flota

**Dashboard en**: `http://localhost:8501`

##### Opcion B: Proyecto 2 - Dashboard de pronostico de ventas

```bash
cd "sales-forecasting"
pip install -r requirements.txt
streamlit run app.py
```

**Que veras**:
- Pronosticos por tienda y producto
- Recomendaciones de inventario
- Metricas de precision
- Drivers de demanda

**Dashboard en**: `http://localhost:8501`

**Nota**: Proyecto 2 incluye datos de pronóstico de muestra (`data_forecast.csv`) para demostración inmediata. El dashboard también genera datos sintéticos automáticamente si el archivo no está disponible.

##### Opcion C: Proyecto 3 - Dashboard de Analisis de Sentimiento Financiero

```bash
cd "financial-sentiment-analysis"
pip install -r requirements.txt
streamlit run app.py
```

**Que veras**:
- Clasificacion de sentimiento financiero en tiempo real (Positivo / Neutro / Negativo)
- Predicciones de FinBERT con puntuaciones de confianza
- Analisis de palabras clave y frases
- Analisis en lote de multiples noticias

**Dashboard en**: `http://localhost:8501`

**Nota**: Proyecto 3 incluye datos de muestra. El dashboard es completamente funcional sin archivos adicionales.

##### Opcion D: Proyecto 4 - Dashboard de Prediccion de Churn Telecom

```bash
cd "customer-churn-prediction"
pip install -r requirements.txt
streamlit run app.py
```

**Que veras**:
- Puntuacion de riesgo de abandono por cliente
- Explicacion de factores de churn basada en SHAP
- Segmentacion de clientes por nivel de riesgo
- Estimacion de impacto financiero y recomendaciones de retencion

**Dashboard en**: `http://localhost:8501`

**Nota**: Proyecto 4 incluye datos de muestra. El dashboard es completamente funcional sin archivos adicionales.

---

### 🔧 Solucion de problemas

#### Problema: "Command 'streamlit' not found"

**Solucion**:
```bash
pip install -r requirements.txt
```

#### Problema: errores "Module not found"

**Solucion**:
```bash
# Proyecto 1
cd "turbofan-predictive-maintenance"
pip install -r requirements.txt

# Proyecto 2
cd "sales-forecasting"
pip install -r requirements.txt
```

#### Problema: el dashboard no abre

**Solucion**: abrir manualmente en el navegador:
```
http://localhost:8501
```

#### Problema: puerto en uso

**Solucion**:
```bash
streamlit run app.py --server.port 8502
```

---

### 📱 Features del dashboard

#### Proyecto 1: Mantenimiento predictivo
- 100% funcional
- Seleccion de motor (1-100)
- 21 sensores
- Prediccion RUL en tiempo real
- Estado de salud (Healthy/Warning/Critical)

#### Proyecto 2: Pronostico de ventas
- Funcional (con archivo de datos)
- Seleccion de tienda (1-54)
- Seleccion de categoria (33 familias)
- Pronostico a 15 dias
- Recomendaciones de inventario

#### Proyecto 3: Analisis de Sentimiento Financiero
- 100% funcional - Datos de demo incluidos
- Clasificacion de texto financiero (Positivo / Neutro / Negativo)
- Puntuaciones de confianza de FinBERT
- Analisis de palabras clave y frases
- Analisis en lote de noticias

#### Proyecto 4: Prediccion de Churn - Telecomunicaciones
- 100% funcional - Datos de demo incluidos
- Puntuacion de riesgo de abandono por cliente
- Explicacion de factores de churn con SHAP
- Clasificacion de riesgo (Bajo/Medio/Alto)
- Estimacion de impacto financiero y retencion
- Predicciones en tiempo real

---

### 💼 Para clientes y reclutadores

Los cuatro dashboards están **desplegados online y listos para producción**. Accede a ellos instantáneamente mediante los enlaces de demo en vivo al inicio de esta guía — sin instalación necesaria.

Demuestran:
1. Analitica en tiempo real
2. Insights accionables
3. Buena experiencia de usuario
4. Habilidades tecnicas en Python y ML

---

### 📧 Soporte

Si hay problemas al abrir los dashboards:

**Franklin Ramos**
- GitHub: [@frankliramos](https://github.com/frankliramos)
- Repositorio: [Proyectos-portafolio](https://github.com/frankliramos/Proyectos-portafolio)

---

### 🎥 Tutoriales en video (proximamente)

- [ ] Walkthrough Dashboard Proyecto 1
- [ ] Walkthrough Dashboard Proyecto 2
- [ ] Guia de instalacion Windows
- [ ] Guia de instalacion Mac/Linux

---

**Ultima actualizacion**: March 2026
