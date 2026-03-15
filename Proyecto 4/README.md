# Customer Churn Prediction - Telecom

## English

### Overview
This project builds a **churn prediction system** for a telecom provider using XGBoost.
It identifies customers at risk, explains key drivers, and estimates financial impact
from retention actions.

### Business Problem
Churn erodes recurring revenue and increases acquisition costs. The model supports
proactive retention by ranking customers and quantifying revenue at risk.

### Data
- Source: Telco Customer Churn dataset (Excel)
- Target: `Churn Value`
- Features: contract, tenure, charges, services, and demographics

### Modeling
- Algorithm: XGBoost (CPU)
- Metrics: Accuracy, Recall, ROC-AUC
- Interpretability: SHAP plots and top drivers

### Results (from notebook)
- Accuracy: 0.93
- Recall (Churn): 0.86
- ROC-AUC: 0.9819

### Project Structure
```
Proyecto 4/
├── dashboard/                 # Streamlit dashboard
│   └── app.py
├── notebooks/                 # EDA and analysis
│   └── 01_eda_initial.ipynb
├── reports/                   # Executive summary
│   └── Executive_Summary.md
└── src/                       # Data + modeling pipeline
    ├── business_metrics.py
    ├── business_value.py
    ├── config.py
    ├── data_loader.py
    ├── modeling.py
    ├── pipeline.py
    └── preprocessing.py
```

### Run the Pipeline
```bash
python -m src.pipeline
```

### Run the Dashboard
```bash
streamlit run dashboard/app.py
```

---

## Espanol

### Resumen
Este proyecto construye un **sistema de prediccion de churn** para telecom usando XGBoost.
Identifica clientes en riesgo, explica drivers y estima impacto financiero.

### Datos
- Fuente: Telco Customer Churn (Excel)
- Target: `Churn Value`
- Features: contrato, antiguedad, cargos, servicios y demografia

### Resultados (segun notebook)
- Accuracy: 0.93
- Recall (Churn): 0.86
- ROC-AUC: 0.9819

### Ejecucion
```bash
python -m src.pipeline
```

### Dashboard
```bash
streamlit run dashboard/app.py
```

## 👤 Author

**Franklin Ramos**
- 📧 Email: franklin.ram.riv@gmail.com
- 💼 LinkedIn: [linkedin.com/in/franklin-ramos-riveros-62b70083](https://www.linkedin.com/in/franklin-ramos-riveros-62b70083/)
- 💼 GitHub: [github.com/frankliramos](https://github.com/frankliramos)
- Portfolio: [GitHub Portfolio](https://github.com/frankliramos/Proyectos-portafolio)
