# Portafolio de Data Science y Machine Learning

**Franklin Ramos**

[🇬🇧 English Version](./README.md)

---

## 📊 Descripción del Portafolio

Este repositorio presenta proyectos profesionales de ciencia de datos y aprendizaje automático, demostrando experiencia en modelado predictivo, deep learning y sistemas de ML listos para producción.

---

## 🚀 Proyecto Destacado: Mantenimiento Predictivo de Turbofan

### [Dashboard Interactivo →](./turbofan-predictive-maintenance)

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

**Sistema de mantenimiento predictivo listo para producción para motores turbofan de aeronaves**

#### 🎯 Aspectos Destacados del Proyecto

- **Impacto de Negocio**: Predice fallas de motores antes de que ocurran, reduciendo mantenimiento no programado en 30-40%
- **Stack Técnico**: Redes neuronales LSTM, PyTorch, dashboard de monitoreo en tiempo real
- **Dataset**: NASA CMAPSS - 100+ motores con trayectorias completas hasta falla
- **Rendimiento**: MAE ~14.2 ciclos, RMSE ~19.7 ciclos, R² 0.78

#### 🔧 Características Principales

✅ **Monitoreo de Salud en Tiempo Real** - Predicciones de RUL (Vida Útil Remanente) en vivo  
✅ **Dashboard Interactivo** - Visualización basada en Streamlit con 21 flujos de sensores  
✅ **Arquitectura de Deep Learning** - LSTM multicapa con regularización dropout  
✅ **Gestión de Flota** - Monitoreo del estado de salud de toda la flota de un vistazo  
✅ **Listo para Producción** - Testing exhaustivo, documentación y manejo de errores completo  

#### 📱 Inicio Rápido

```bash
cd turbofan-predictive-maintenance
pip install -r requirements.txt
streamlit run app.py
```

El dashboard se lanzará en `http://localhost:8501`

#### 📖 Documentación Completa

- [🇬🇧 Documentación en Inglés](./turbofan-predictive-maintenance/README.md)
- [🇪🇸 Documentación en Español](./turbofan-predictive-maintenance/README_ES.md)
- [📊 Ficha Técnica del Modelo](./turbofan-predictive-maintenance/MODEL_CARD.md)
- [⚡ Guía de Inicio Rápido](./turbofan-predictive-maintenance/QUICKSTART.md)

---

## 🛠️ Habilidades Técnicas Demostradas

### Machine Learning y Deep Learning
- **Pronóstico de Series Temporales** - Redes LSTM para datos secuenciales
- **Ingeniería de Características** - Preprocesamiento y normalización de datos de sensores
- **Optimización de Modelos** - Ajuste de hiperparámetros, early stopping
- **Evaluación de Modelos** - Métricas MAE, RMSE, R² con estrategias de validación

### Ingeniería de Software
- **Código de Producción** - Arquitectura modular, manejo de errores, logging
- **Desarrollo de Dashboard** - Aplicaciones interactivas con Streamlit
- **Pipeline de Datos** - Procesos ETL para dataset NASA CMAPSS
- **Testing** - Pruebas unitarias y validación de datos
- **Documentación** - Documentación técnica y de usuario exhaustiva

### Herramientas y Tecnologías
- **Lenguajes**: Python 3.12+
- **Frameworks ML**: PyTorch, scikit-learn
- **Procesamiento de Datos**: Pandas, NumPy
- **Visualización**: Matplotlib, Seaborn, Streamlit
- **Desarrollo**: Git, Docker, Jupyter

---

## 📊 Métricas del Proyecto

| Métrica | Valor |
|---------|-------|
| **Líneas de Código** | 2,500+ |
| **Cobertura de Tests** | Validación de datos comprehensiva |
| **Páginas de Documentación** | 5 documentos técnicos |
| **Datos Procesados** | 33,727 ciclos en 200 motores |
| **Precisión del Modelo** | R² = 0.78 |

---

## 🎯 Objetivos Profesionales

Este portafolio demuestra:

- Capacidad para traducir problemas de negocio en soluciones ML
- Ejecución de proyectos ML de extremo a extremo desde EDA hasta despliegue
- Código listo para producción con estándares profesionales
- Documentación clara y habilidades de comunicación
- Comprensión de aspectos técnicos y de negocio

---

## 📬 Contacto

**Franklin Ramos**

- 📧 Email: Disponible bajo solicitud
- 💼 GitHub: [github.com/frankliramos](https://github.com/frankliramos)
- 🌐 Portafolio: Este repositorio

---

## 📄 Licencia

Este proyecto está disponible para fines educativos y de revisión de portafolio. Ver directorios de proyectos individuales para información específica de licencia.

---

## 🔍 Estructura del Repositorio

```
Proyectos-portafolio/
├── README.md                           # Versión en inglés
├── README_ES.md                        # Este archivo (Español)
└── turbofan-predictive-maintenance/   # Proyecto de mantenimiento predictivo
    ├── app.py                         # Aplicación dashboard
    ├── README.md                      # Documentación del proyecto (EN)
    ├── README_ES.md                   # Documentación del proyecto (ES)
    ├── MODEL_CARD.md                  # Especificaciones del modelo
    ├── QUICKSTART.md                  # Guía de inicio rápido
    ├── requirements.txt               # Dependencias
    ├── data/                          # Dataset NASA CMAPSS
    ├── models/                        # Modelos entrenados
    ├── notebooks/                     # Jupyter notebooks
    ├── src/                           # Código fuente
    └── results/                       # Resultados del modelo
```

---

**Última Actualización**: Febrero 2026
