# Portafolio de Data Science & Machine Learning

**Franklin Ramos**

[🇬🇧 English Version](./README.md)

---

## 📊 Visión General del Portafolio

Bienvenido a mi portafolio de Data Science y Machine Learning. Este repositorio presenta proyectos profesionales end-to-end que demuestran experiencia en modelado predictivo, deep learning, pronóstico de series temporales y sistemas ML listos para producción.

### 🖥️ Dashboards Interactivos

Todos los proyectos incluyen **dashboards interactivos de Streamlit** para visualización y exploración en tiempo real:

- **Proyecto 1**: Monitoreo de salud de motores con predicciones de RUL
- **Proyecto 2**: Pronóstico de ventas con recomendaciones de inventario
- **Proyecto 3**: Evaluación de riesgo de abandono de clientes y estrategias de retención

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

[📂 Ver Proyecto →](./Proyecto%201/turbofan-predictive-maintenance)

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

[📂 Ver Proyecto →](./Proyecto%202)

**Características Principales**:
- Pronóstico de ventas a 15 días por tienda y familia de producto
- Dashboard interactivo con predicciones en tiempo real
- Recomendaciones de optimización de inventario
- Integración de factores externos (precios del petróleo, feriados)
- Documentación completa (Inglés y Español)

---

### Proyecto 3: Sistema de Predicción de Abandono de Clientes
**Estado**: ✅ Completo | **Tipo**: Clasificación, Analítica de Clientes, Banca

Sistema avanzado de predicción de abandono de clientes para instituciones bancarias usando ensemble de machine learning.

- **Impacto de Negocio**: 25-35% reducción de abandono, $11M+ ahorro anual, 650% ROI
- **Stack Tecnológico**: XGBoost, Random Forest, SMOTE, Dashboard Streamlit
- **Dataset**: 10,000 clientes con demografía y comportamiento bancario
- **Rendimiento**: 86.5% precisión, F1-Score 82.7%, ROC-AUC 0.91

[📂 Ver Proyecto →](./Proyecto%203)

**Características Clave**:
- Evaluación de riesgo de abandono en tiempo real (individual y lotes)
- Dashboard de segmentación de clientes interactivo
- Recomendaciones de estrategia de retención personalizadas
- Interpretabilidad del modelo basada en SHAP
- Documentación completa (Inglés y Español)

---

### Proyecto 4: Próximamente
**Estado**: 🔜 En Planificación

[📂 Ver Proyecto →](./Proyecto%204)

---

## 🛠️ Habilidades Técnicas Demostradas

### Machine Learning & Deep Learning
- **Pronóstico de Series Temporales** - LSTM, XGBoost, patrones estacionales
- **Clasificación** - Métodos ensemble, manejo de datos desbalanceados (SMOTE)
- **Ingeniería de Características** - Datos de sensores, métricas retail, comportamiento de clientes, conocimiento del dominio
- **Optimización de Modelos** - Ajuste de hiperparámetros, validación cruzada
- **Evaluación de Modelos** - MAE, RMSE, R², F1-Score, ROC-AUC, métricas de negocio
- **Interpretabilidad de Modelos** - Valores SHAP, importancia de características

### Ingeniería de Software
- **Código de Producción** - Arquitectura modular, manejo de errores, logging
- **Desarrollo de Dashboards** - Aplicaciones Streamlit interactivas
- **Pipelines de Datos** - Procesos ETL, validación de datos
- **Testing** - Pruebas unitarias, pruebas de integración, verificación de calidad
- **Documentación** - Documentos técnicos, guías de usuario, model cards

### Herramientas & Tecnologías
- **Lenguajes**: Python 3.12+
- **Frameworks ML**: PyTorch, scikit-learn, XGBoost, imbalanced-learn
- **Procesamiento de Datos**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn, Plotly, Streamlit
- **Interpretabilidad**: SHAP
- **Desarrollo**: Git, Docker, Jupyter

---

## 📊 Métricas del Portafolio

| Métrica | Valor |
|---------|-------|
| **Proyectos Totales** | 4 (3 completos, 1 planificado) |
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
├── .gitignore                               # Gitignore global
│
├── Proyecto 1/                              # Mantenimiento Predictivo Turbofán
│   └── turbofan-predictive-maintenance/
│       ├── app.py                           # Dashboard interactivo
│       ├── README.md                        # Documentación del proyecto
│       ├── data/                            # Dataset NASA CMAPSS
│       ├── models/                          # Modelos entrenados
│       ├── notebooks/                       # Análisis Jupyter
│       ├── src/                             # Código fuente
│       └── results/                         # Evaluación de modelos
│
├── Proyecto 2/                              # Pronóstico de Ventas (Próximamente)
│   ├── dashboard/                           # Aplicación Streamlit
│   ├── data/                                # Datos retail
│   ├── models/                              # Modelos XGBoost
│   ├── notebooks/                           # EDA y modelado
│   └── src/                                 # Código fuente
│
├── Proyecto 3/                              # Próximamente
│   └── README.md
│
└── Proyecto 4/                              # Próximamente
    └── README.md
```

---

## 🚀 Inicio Rápido

### Ejecutar Dashboard del Proyecto 1

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd "Proyectos-portafolio/Proyecto 1/turbofan-predictive-maintenance"
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

**Última Actualización**: Febrero 2026
