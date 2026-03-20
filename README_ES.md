# Portafolio de Data Science & Machine Learning

**Franklin Ramos**

[![CI](https://github.com/frankliramos/Proyectos-portafolio/actions/workflows/ci.yml/badge.svg)](https://github.com/frankliramos/Proyectos-portafolio/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-yellow.svg)](./LICENSE)

[🇬🇧 English Version](./README.md)

---

## 📊 Visión General del Portafolio

Bienvenido a mi portafolio de Data Science y Machine Learning. Este repositorio presenta proyectos profesionales end-to-end que demuestran experiencia en modelado predictivo, deep learning, pronóstico de series temporales y sistemas ML listos para producción.

### 🖥️ Dashboards Interactivos

Todos los proyectos incluyen **dashboards interactivos de Streamlit** para visualización y exploración en tiempo real:

- **Proyecto 1**: Monitoreo de salud de motores con predicciones de RUL
- **Proyecto 2**: Pronóstico de ventas con recomendaciones de inventario
- **Proyecto 3**: Evaluación de riesgo de abandono de clientes y estrategias de retención
- **Proyecto 4**: Recomendaciones de productos y analíticas de e-commerce

**Inicio Rápido**:
```bash
# Para cualquier proyecto con dashboard
cd "Proyecto X/[directorio-del-proyecto]"
pip install -r requirements.txt
streamlit run app.py
```

📖 **[Guía Completa de Acceso a Dashboards →](./DASHBOARD_ACCESS.md)**

---

## 🚀 Proyectos

### Proyecto 1: Mantenimiento Predictivo de Turbofán
**Estado**: ✅ Completo | **Tipo**: Deep Learning, Series Temporales, Mantenimiento Predictivo

Sistema de mantenimiento predictivo listo para producción para motores turbofán de aeronaves usando redes neuronales LSTM.

- **Impacto de Negocio**: Predice fallos de motor con 30-40 ciclos de anticipación
- **Stack Tecnológico**: PyTorch, LSTM, Dashboard Streamlit
- **Dataset**: NASA CMAPSS (200 motores, 33K+ ciclos)
- **Rendimiento**: MAE ~14.2 ciclos, RMSE ~19.7, R² 0.78

[📂 Ver Proyecto →](./turbofan-predictive-maintenance)

**Características Clave**:
- Predicciones de RUL (Vida Útil Restante) en tiempo real
- Dashboard interactivo con 21 flujos de sensores
- Monitoreo de gestión de flota
- Documentación completa (Inglés y Español)

---

### Proyecto 2: Sistema de Pronóstico de Ventas
**Estado**: ✅ Completo | **Tipo**: Series Temporales, XGBoost, Pronóstico de Demanda

Sistema avanzado de pronóstico de ventas minoristas para tiendas ecuatorianas usando XGBoost con aceleración GPU.

- **Impacto de Negocio**: 83% de precisión en predicciones (WAPE 16.9%), 15-20% reducción en costos de inventario
- **Stack Tecnológico**: XGBoost GPU, Dashboard Streamlit, Pandas
- **Dataset**: 2.9M+ transacciones, 54 tiendas, 33 categorías de productos
- **Rendimiento**: RMSLE 0.40, WAPE 16.9%

[📂 Ver Proyecto →](./sales-forecasting)

**Características Principales**:
- Pronóstico de ventas a 15 días por tienda y familia de producto
- Dashboard interactivo con predicciones en tiempo real
- Recomendaciones de optimización de inventario
- Integración de factores externos (precios del petróleo, feriados)
- Documentación completa (Inglés y Español)

---

### Proyecto 3: Análisis de Sentimiento Financiero
**Estado**: ✅ Completo | **Tipo**: PLN, FinBERT, Analítica Financiera

Sistema avanzado de predicción de abandono de clientes para instituciones bancarias usando ensemble de machine learning.

- **Impacto de Negocio**: 25-35% reducción de abandono, $11M+ ahorro anual, 650% ROI
- **Stack Tecnológico**: XGBoost, Random Forest, SMOTE, Dashboard Streamlit
- **Dataset**: 10,000 clientes con demografía y comportamiento bancario
- **Rendimiento**: 86.5% precisión, F1-Score 82.7%, ROC-AUC 0.91

[📂 Ver Proyecto →](./financial-sentiment-analysis)

**Características Clave**:
- Evaluación de riesgo de abandono en tiempo real (individual y lotes)
- Dashboard de segmentación de clientes interactivo
- Recomendaciones de estrategia de retención personalizadas
- Interpretabilidad del modelo basada en SHAP
- Documentación completa (Inglés y Español)

---

### Proyecto 4: Predicción de Abandono de Clientes Telecom
**Estado**: ✅ Completo | **Tipo**: Clasificación, Analítica de Clientes, Telecomunicaciones

Motor de recomendación híbrido avanzado que combina filtrado colaborativo y basado en contenido para personalización de comercio electrónico.

- **Impacto de Negocio**: 20-30% aumento en conversión, 85% aumento de ingresos por usuario, 2x CTR
- **Stack Tecnológico**: Filtrado Colaborativo (ALS), Basado en Contenido (TF-IDF), Modelo Híbrido, Streamlit
- **Dataset**: 50,000+ usuarios, 10,000+ productos, 500,000+ interacciones
- **Rendimiento**: Precision@10: 0.341, NDCG@10: 0.412, ROI: 1,567%-2,433%

[📂 Ver Proyecto →](./customer-churn-prediction)

**Características Clave**:
- Recomendaciones personalizadas de productos con puntajes de confianza
- Motor de descubrimiento de productos similares
- Soporte multi-algoritmo (Colaborativo, Basado en Contenido, Híbrido, Neural CF)
- Dashboard interactivo con recomendaciones en tiempo real
- Framework de pruebas A/B y analíticas
- Documentación completa (Inglés y Español)

---

## 🛠️ Habilidades Técnicas Demostradas

### Machine Learning & Deep Learning
- **Pronóstico de Series Temporales** - LSTM, XGBoost, patrones estacionales
- **Clasificación** - Métodos ensemble, manejo de datos desbalanceados (SMOTE)
- **Sistemas de Recomendación** - Filtrado colaborativo, filtrado basado en contenido, modelos híbridos
- **Ingeniería de Características** - Datos de sensores, métricas retail, comportamiento de clientes, conocimiento del dominio
- **Optimización de Modelos** - Ajuste de hiperparámetros, validación cruzada
- **Evaluación de Modelos** - MAE, RMSE, R², F1-Score, ROC-AUC, Precision@K, NDCG, métricas de negocio
- **Interpretabilidad de Modelos** - Valores SHAP, importancia de características

### Ingeniería de Software
- **Código de Producción** - Arquitectura modular, manejo de errores, logging
- **Desarrollo de Dashboards** - Aplicaciones Streamlit interactivas
- **Pipelines de Datos** - Procesos ETL, validación de datos
- **Testing** - Pruebas unitarias, pruebas de integración, verificación de calidad
- **Documentación** - Documentos técnicos, guías de usuario, model cards

### Herramientas & Tecnologías
- **Lenguajes**: Python 3.12+
- **Frameworks ML**: PyTorch, scikit-learn, XGBoost, TensorFlow, imbalanced-learn
- **Recomendación**: Implicit, Surprise, LightFM
- **NLP**: NLTK, spaCy, TF-IDF
- **Procesamiento de Datos**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn, Plotly, Streamlit
- **Interpretabilidad**: SHAP
- **Desarrollo**: Git, Docker, Jupyter

---

## 📊 Métricas del Portafolio

| Métrica | Valor |
|---------|-------|
| **Proyectos Totales** | 4 completados |
| **Líneas de Código** | 8,000+ |
| **Páginas de Documentación** | 20+ documentos técnicos |
| **Tecnologías** | 15+ frameworks y herramientas |
| **Docs Bilingües** | Inglés y Español |

---

## 🎯 Enfoque Profesional

Este portafolio demuestra:

1. **Valor de Negocio** - Resolviendo problemas del mundo real con impacto medible
2. **Excelencia Técnica** - Código listo para producción con mejores prácticas
3. **Comunicación** - Documentación clara para audiencias técnicas y no técnicas
4. **Ejecución End-to-End** - Desde definición del problema hasta despliegue
5. **Escalabilidad** - Estructura organizada para múltiples proyectos

---

## 🔍 Estructura del Repositorio

```
Proyectos-portafolio/
├── README.md                                # Versión en inglés
├── README_ES.md                             # Este archivo (Español)
├── LICENSE                                  # Licencia MIT
├── .gitignore                               # Gitignore global
├── .github/workflows/ci.yml                 # Pipeline CI/CD
│
├── turbofan-predictive-maintenance/         # Mantenimiento Predictivo Turbofán
│   ├── app.py                           # Dashboard interactivo (entrada Streamlit)
│   ├── README.md                        # Documentación del proyecto
│   ├── MODEL_CARD.md                    # Especificaciones del modelo
│   ├── data/                            # Dataset NASA CMAPSS (raw + procesado)
│   ├── models/                          # Modelos LSTM entrenados
│   ├── notebooks/                       # Notebooks Jupyter EDA y modelado
│   ├── src/                             # Módulos de código fuente
│   ├── results/                         # Resultados de evaluación
│   ├── dashboard/                       # Fuente del dashboard
│   └── Dockerfile                       # Definición del contenedor
│
├── sales-forecasting/                       # Pronóstico de Ventas
│   ├── app.py                           # Entrada Streamlit
│   ├── dashboard/app.py                 # Fuente del dashboard
│   ├── notebooks/                       # Notebooks EDA y modelado
│   ├── src/                             # Ingeniería de características y predicción
│   └── requirements.txt                 # Dependencias
│
├── financial-sentiment-analysis/            # Análisis de Sentimiento Financiero (FinBERT)
│   ├── app.py                           # Dashboard interactivo (entrada Streamlit)
│   ├── README.md                        # Documentación del proyecto
│   ├── data/                            # Dataset Financial PhraseBank
│   ├── models/                          # Modelos FinBERT entrenados
│   ├── notebooks/                       # Notebooks de análisis
│   ├── src/                             # Módulos de código fuente
│   └── results/                         # Resultados de evaluación
│
└── customer-churn-prediction/               # Predicción de Abandono de Clientes Telecom
    ├── app.py                           # Dashboard interactivo (entrada Streamlit)
    ├── README.md                        # Documentación del proyecto
    ├── data/                            # Dataset de clientes telecom
    ├── models/                          # Modelos XGBoost entrenados
    ├── notebooks/                       # Notebooks de análisis
    ├── src/                             # Módulos de código fuente
    └── results/                         # Resultados de evaluación
```

---

## 🚀 Inicio Rápido

### Ejecutar Dashboard de Mantenimiento Predictivo Turbofán

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd "Proyectos-portafolio/turbofan-predictive-maintenance"
pip install -r requirements.txt
streamlit run app.py
```

### Ejecutar Dashboard de Pronóstico de Ventas

```bash
cd "Proyectos-portafolio/sales-forecasting"
pip install -r requirements.txt
streamlit run app.py
```

### Ejecutar Dashboard de Análisis de Sentimiento Financiero

```bash
cd "Proyectos-portafolio/financial-sentiment-analysis"
pip install -r requirements.txt
streamlit run app.py
```

### Ejecutar Dashboard de Predicción de Abandono de Clientes

```bash
cd "Proyectos-portafolio/customer-churn-prediction"
pip install -r requirements.txt
streamlit run app.py
```

El dashboard se abre en `http://localhost:8501`

---

## 📬 Contacto

**Franklin Ramos**

- 📧 Email: Disponible bajo petición
- 💼 GitHub: [github.com/frankliramos](https://github.com/frankliramos)
- 🌐 Portafolio: Este repositorio

---

## 📄 Licencia

Este proyecto está disponible para propósitos educativos y revisión de portafolio. Ver directorios de proyectos individuales para información específica de licencia.

---

**Última Actualización**: Marzo 2026
