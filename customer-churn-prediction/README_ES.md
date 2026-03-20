# Prediccion de Churn - Telecom

## Espanol

### Resumen
Este proyecto implementa un **sistema de prediccion de churn** con XGBoost para
telecomunicaciones. Identifica clientes en riesgo, explica drivers y estima
impacto financiero.

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

---

## English

### Overview
This project delivers a **churn prediction system** with XGBoost. It identifies
customers at risk, explains key drivers, and estimates financial impact.

### Results (from notebook)
- Accuracy: 0.93
- Recall (Churn): 0.86
- ROC-AUC: 0.9819
