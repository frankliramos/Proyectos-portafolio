# 📊 Sistema de Predicción de Churn para Telecomunicaciones
## Reporte Ejecutivo

---

### ENGLISH SUMMARY

The churn model identifies high-risk customers to enable proactive retention,
with a **projected annual savings of $74,076 USD** (assuming 40% retention).

**Key metrics:**
- Accuracy: 80%
- Recall (Churn): 55%
- Precision (Churn): 64%
- ROC-AUC: 0.8448

*Evaluated on a 20% stratified hold-out set, after removing target-leaking columns (Churn Score, CLTV, Churn Label, Churn Reason).*

**Main churn reasons stated by churned customers (descriptive, not model output):**
1. Support staff attitude
2. Competitor higher download speeds
3. Competitor offers more data

**Top model features (SHAP):** contract type, dependents, tenure, monthly charges.

**Business takeaway:** Service quality and competitor offers outweigh price in the reasons customers give for leaving.

---

### 🎯 RESUMEN EJECUTIVO

El modelo predictivo desarrollado identifica clientes con alto riesgo de cancelar su servicio, permitiendo acciones de retención proactivas que generan un **ahorro anual proyectado de $74,076 USD**.

---

### 💼 PROBLEMA DE NEGOCIO

La empresa presenta una **tasa de churn del 26.54%**, significativamente superior al promedio saludable de la industria (10-15%). Cada cliente perdido representa:

- **Pérdida inmediata de ingresos mensuales recurrentes**
- **Costo de adquisición de cliente (CAC) no recuperado**
- **Deterioro del valor de marca y reputación**

**Pérdida mensual actual estimada (todos los clientes que se fueron):** $139,130.85

---

### ✅ SOLUCIÓN IMPLEMENTADA

Sistema de Machine Learning basado en XGBoost que:

1. **Identifica clientes en riesgo** con 55% de recall (detecta correctamente 55 de cada 100 clientes que están por irse)
2. **Segmenta por nivel de riesgo** (Bajo, Medio, Alto) para priorizar acciones
3. **Explica las causas** de abandono mediante SHAP values
4. **Cuantifica el impacto financiero** en tiempo real

---

### 💰 RETORNO DE INVERSIÓN (ROI)

| Métrica | Valor |
|---------|-------|
| Ingresos Mensuales en Riesgo Identificados (churners detectados en el test) | **$15,432.60** |
| Ahorro Mensual (Retención 40%) | **$6,173.04** |
| **AHORRO ANUAL PROYECTADO** | **$74,076.48** |

*Asumiendo una tasa de retención del 40% mediante campañas dirigidas.*

---

### 🔍 DRIVERS PRINCIPALES DE CHURN

Motivos de abandono declarados por los clientes que se fueron (análisis descriptivo del dataset, no salida del modelo) (Top 3):

1. **Actitud del personal de soporte (10.3%)** → Acción: Capacitación y mejora de servicio al cliente
2. **Competencia ofrece mayor velocidad de descarga (10.1%)** → Acción: Upgrade de infraestructura o comunicación de valor
3. **Competencia ofrece más datos (8.7%)** → Acción: Revisión de planes y ofertas competitivas

**Insight clave:** El precio NO es el principal driver. El servicio y la experiencia tienen mayor impacto.

---

### 📈 RENDIMIENTO DEL MODELO

| Métrica | Valor |
|---------|-------|
| Accuracy | 80% |
| Recall (Clase Churn) | **55%** |
| Precision (Clase Churn) | 64% |
| ROC-AUC | **0.8448** |

**Traducción a negocio:** De cada 100 clientes que están por irse, el modelo identifica correctamente a 55; de cada 100 que marca como churn, 64 realmente se van.

---

### 🎯 SEGMENTACIÓN DE RIESGO

Distribución de clientes en el set de prueba:

- **Alto Riesgo:** 161 clientes → Acción inmediata (llamada personalizada, oferta especial)
- **Riesgo Medio:** 358 clientes → Monitoreo + campaña email
- **Bajo Riesgo:** 890 clientes → Estrategia de fidelización estándar

---

### 📊 VARIABLES MÁS IMPORTANTES (SHAP Analysis)

1. **Tipo de Contrato** (Month-to-month = Mayor riesgo)
2. **Dependents** (Sin dependientes = Mayor riesgo)
3. **Tenure Months** (Clientes nuevos = Mayor riesgo)
4. **Monthly Charges** (Cargos altos = Mayor riesgo)
5. **Total Charges**

---

### 🚀 ESTRATEGIA DE RETENCIÓN RECOMENDADA

#### Para clientes de **Alto Riesgo:**
- Contacto telefónico personalizado dentro de 48 hrs
- Oferta de upgrade a contrato anual con descuento (reducir barrera de salida)
- Soporte técnico prioritario

#### Para clientes de **Riesgo Medio:**
- Campaña de email marketing con beneficios exclusivos
- Encuesta de satisfacción para detectar fricciones

#### Para clientes de **Bajo Riesgo:**
- Programa de fidelización (puntos, beneficios)
- Cross-selling de servicios adicionales

---

### 🔧 IMPLEMENTACIÓN TÉCNICA

**Arquitectura:**
- Pipeline modular en Python (src/)
- Modelo: XGBoost (400 estimadores)
- Interpretabilidad: SHAP values
- Ejecución: `python -m src.pipeline`

**Escalabilidad:**
- Estructura aplicable a SaaS, Banca, Seguros

- Pipeline listo para integración con sistemas CRM
- Posibilidad de automatización con Apache Airflow

---