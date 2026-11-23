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

### Resultados (test hold-out 20%, sin fuga de datos)
- Accuracy: 0.80
- Recall (Churn): 0.55
- ROC-AUC: 0.8448

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

### Results (hold-out test 20%, leakage-free)
- Accuracy: 0.80
- Recall (Churn): 0.55
- ROC-AUC: 0.8448
