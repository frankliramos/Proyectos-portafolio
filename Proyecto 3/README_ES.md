# 💳 Predicción de Abandono de Clientes: Sistema de Retención Bancaria

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)

[🇬🇧 English Version](./README.md)

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema de predicción de abandono de clientes** para instituciones bancarias utilizando técnicas avanzadas de aprendizaje automático. El objetivo es identificar clientes con alto riesgo de abandonar el banco, permitiendo estrategias proactivas de retención y reduciendo la deserción de clientes.

### 🎯 Problema de Negocio

El abandono de clientes en el sector bancario resulta en:
- Pérdida de ingresos por comisiones y transacciones
- Disminución del valor de vida del cliente (CLV)
- Altos costos de adquisición para reemplazar clientes perdidos
- Daño a la reputación de marca y participación de mercado
- Pérdida de oportunidades de venta cruzada y adicional

**Solución**: Predecir qué clientes tienen probabilidad de abandonar con más del 86% de precisión, permitiendo campañas de retención dirigidas que pueden reducir el abandono en un 25-35%.

### 🔬 Enfoque Técnico

- **Modelo**: Ensemble de XGBoost, Random Forest y Regresión Logística
- **Métrica de Optimización**: F1-Score y ROC-AUC
- **Características**: Más de 20 atributos de clientes incluyendo demografía, actividad de cuenta y uso de productos
- **Salida**: Probabilidad de abandono (0-100%) con clasificación de riesgo
- **Manejo de Desbalance de Clases**: Sobremuestreo SMOTE + pesos de clase

## 📊 Dataset

### Datos de Clientes Bancarios

El dataset contiene información comprensiva de clientes de un banco europeo:

- **Clientes**: 10,000 clientes bancarios
- **Características**: 14 atributos cubriendo demografía, productos bancarios y actividad de cuenta
- **Objetivo**: Clasificación binaria (Abandonó: 1, Retenido: 0)
- **Distribución de Clases**: ~20% tasa de abandono (escenario de desbalance realista)

**Características Clave**:
- `customer_id`: Identificador único del cliente
- `credit_score`: Puntaje de crédito (300-850)
- `geography`: País del cliente (Francia, España, Alemania)
- `gender`: Masculino/Femenino
- `age`: Edad del cliente
- `tenure`: Años como cliente del banco
- `balance`: Saldo de la cuenta
- `num_of_products`: Número de productos bancarios utilizados (1-4)
- `has_cr_card`: Tiene tarjeta de crédito (0/1)
- `is_active_member`: Estado de miembro activo (0/1)
- `estimated_salary`: Salario anual estimado
- `exited`: Abandonó (1) o Retenido (0) - **Variable Objetivo**

**Fuente de Datos**: Kaggle Bank Customer Churn Dataset (simulado pero realista)

## 🏗️ Estructura del Proyecto

```
Proyecto 3/
├── app.py                          # Aplicación dashboard Streamlit
├── README.md                       # Versión en inglés
├── README_ES.md                    # Este archivo
├── requirements.txt                # Dependencias de Python
├── data/
│   ├── raw/                        # Dataset original
│   │   └── bank_churn.csv
│   └── processed/                  # Datos preprocesados
│       └── churn_prepared.parquet
├── models/                         # Modelos entrenados y artefactos
│   ├── xgboost_model.pkl          # Clasificador XGBoost
│   ├── random_forest_model.pkl    # Clasificador Random Forest
│   ├── ensemble_model.pkl         # Clasificador ensemble voting
│   ├── scaler.pkl                 # StandardScaler para características
│   └── feature_names.pkl          # Nombres de columnas de características
├── notebooks/                      # Notebooks Jupyter para análisis
│   ├── 01_eda_churn.ipynb         # Análisis Exploratorio de Datos
│   ├── 02_feature_engineering.ipynb  # Creación de características
│   ├── 03_model_baseline.ipynb    # Modelos base
│   └── 04_model_ensemble.ipynb    # Entrenamiento de modelo ensemble
├── results/                        # Resultados de evaluación de modelo
│   ├── metrics_comparison.csv
│   ├── feature_importance.csv
│   └── confusion_matrix.png
└── src/                            # Módulos de código fuente
    ├── __init__.py
    ├── config.py                   # Configuración y rutas
    ├── data_loader.py              # Utilidades de carga de datos
    ├── preprocessing.py            # Funciones de preprocesamiento
    ├── feature_engineering.py      # Ingeniería de características
    ├── models.py                   # Funciones de entrenamiento de modelos
    └── inference.py                # Motor de predicción
```

## 🚀 Comenzando

### Prerequisitos

- Python 3.10 o superior
- Gestor de paquetes pip

### Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd "Proyectos-portafolio/Proyecto 3"
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

4. **Verificar archivos de datos**: Asegurar que el directorio `data/raw/` contenga el dataset.

### Ejecutar el Dashboard

Lanzar el dashboard interactivo de Streamlit:

```bash
streamlit run app.py
```

El dashboard se abrirá en su navegador en `http://localhost:8501`.

## 📱 Dashboard Interactivo

### 🌐 Visualizando el Dashboard

El proyecto incluye un **dashboard interactivo de Streamlit** para evaluación de riesgo de abandono en tiempo real e insights de clientes.

**Acceso Rápido**:
```bash
# Desde el directorio Proyecto 3
streamlit run app.py
```

El dashboard se abre automáticamente en `http://localhost:8501` y proporciona:
- Evaluación de riesgo de abandono de cliente individual
- Segmentación de clientes por nivel de riesgo
- Visualización de importancia de características
- Recomendaciones de estrategias de retención
- Capacidades de predicción por lotes

![Dashboard de Abandono de Clientes](../assets/proyecto3-dashboard.png)

### Características del Dashboard

### 1. **Predicción Individual de Abandono**
- Ingresar detalles del cliente (edad, saldo, antigüedad, etc.)
- Obtener probabilidad de abandono en tiempo real (0-100%)
- Clasificación de riesgo (🟢 Bajo | 🟡 Medio | 🔴 Alto)
- Recomendaciones de retención personalizadas

### 2. **Segmentación de Clientes**
- Ver todos los clientes por nivel de riesgo
- Filtrar por demografía y atributos de cuenta
- Ordenar por probabilidad de abandono
- Exportar listas de clientes de alto riesgo

### 3. **Análisis de Características**
- Visualización de importancia de características
- Valores SHAP para interpretabilidad del modelo
- Comprensión de impulsores clave de abandono
- Patrones demográficos y de comportamiento

### 4. **Estrategias de Retención**
- Motor de recomendaciones automatizado
- Acciones de retención personalizadas por nivel de riesgo
- ROI esperado de campañas de retención
- Priorización de campañas

### Opciones de Configuración

**Controles de Barra Lateral**:
- Selección/ingreso de ID de cliente
- Ajuste de umbral de riesgo (Bajo/Medio/Alto)
- Selección de modelo (XGBoost, Random Forest, Ensemble)
- Filtrado de características
- Opciones de exportación

## 🧠 Arquitectura del Modelo

### Enfoque Ensemble

```python
Modelos:
1. Clasificador XGBoost
   - Profundidad de árbol: 5
   - Tasa de aprendizaje: 0.1
   - N_estimadores: 200
   - Scale_pos_weight: 4.0 (para desbalance de clases)

2. Clasificador Random Forest
   - N_estimadores: 200
   - Profundidad máxima: 15
   - Min muestras división: 10
   - Peso de clase: balanceado

3. Regresión Logística
   - C: 0.1 (regularización)
   - Penalización: L2
   - Peso de clase: balanceado

Estrategia Ensemble: Votación Suave (promedio ponderado de probabilidades)
```

**¿Por qué Ensemble?**
- Combina fortalezas de diferentes algoritmos
- Predicciones más robustas que un solo modelo
- Mejor generalización a nuevos datos
- Riesgo reducido de sobreajuste

### Métricas de Rendimiento

| Métrica | XGBoost | Random Forest | Ensemble |
|---------|---------|---------------|----------|
| **Precisión** | 85.2% | 84.1% | 86.5% |
| **Precision** | 82.3% | 80.7% | 84.1% |
| **Recall** | 78.5% | 79.2% | 81.3% |
| **F1-Score** | 80.3% | 79.9% | 82.7% |
| **ROC-AUC** | 0.89 | 0.88 | 0.91 |

**Métricas de Negocio**:
- **Reducción de Abandono**: 25-35% con retención dirigida
- **ROI**: 3.5x en campañas de retención
- **Tasa de Falsos Positivos**: 12% (bajo costo de predicciones incorrectas)

## 🔧 Entrenamiento del Modelo

### Preprocesamiento de Datos

1. **Manejo de Valores Faltantes**: Estrategia de imputación para características dispersas
2. **Escalado de Características**: StandardScaler para características numéricas
3. **Codificación**: Codificación one-hot para variables categóricas (Geografía, Género)
4. **Desbalance de Clases**: Sobremuestreo SMOTE + pesos de clase
5. **División Train/Test**: 80/20 con estratificación

### Proceso de Entrenamiento

Ejecutar los notebooks en orden:

1. **EDA**: `notebooks/01_eda_churn.ipynb`
   - Análisis de demografía de clientes
   - Patrones y tendencias de abandono
   - Análisis de correlación de características
   - Insights iniciales

2. **Ingeniería de Características**: `notebooks/02_feature_engineering.ipynb`
   - Creación de nuevas características (ej., balance_to_salary_ratio)
   - Interacciones de características
   - Selección de características

3. **Modelos Base**: `notebooks/03_model_baseline.ipynb`
   - Regresión Logística
   - Árboles de Decisión
   - Ajuste de hiperparámetros
   - Validación cruzada

4. **Entrenamiento Ensemble**: `notebooks/04_model_ensemble.ipynb`
   - Entrenamiento de XGBoost y Random Forest
   - Creación de modelo ensemble
   - Evaluación y comparación de modelos
   - Selección del modelo final

## 📈 Ejemplos de Uso

### API Python

```python
from src.inference import ChurnPredictor
from src.data_loader import load_customer_data
from pathlib import Path

# Inicializar motor de predicción
project_root = Path(__file__).parent
predictor = ChurnPredictor(project_root)

# Cargar datos del cliente
customer_data = {
    'credit_score': 650,
    'geography': 'France',
    'gender': 'Female',
    'age': 42,
    'tenure': 5,
    'balance': 125000,
    'num_of_products': 2,
    'has_cr_card': 1,
    'is_active_member': 1,
    'estimated_salary': 75000
}

# Predecir probabilidad de abandono
churn_prob = predictor.predict_proba(customer_data)
print(f"Probabilidad de Abandono: {churn_prob:.1%}")

# Obtener clasificación de riesgo
risk_level = predictor.classify_risk(churn_prob)
print(f"Nivel de Riesgo: {risk_level}")
```

### Predicciones por Lotes

```python
import pandas as pd

# Cargar base de datos de clientes
customers_df = pd.read_csv('data/customer_database.csv')

# Predecir para todos los clientes
predictions = predictor.predict_batch(customers_df)

# Agregar predicciones al DataFrame
customers_df['churn_probability'] = predictions
customers_df['risk_level'] = customers_df['churn_probability'].apply(
    predictor.classify_risk
)

# Identificar clientes de alto riesgo
high_risk = customers_df[customers_df['risk_level'] == 'High']
high_risk.to_csv('high_risk_customers.csv', index=False)

print(f"Clientes de alto riesgo: {len(high_risk)}")
```

## 🔍 Insights Clave

### Importancia de Características

**Principales Predictores de Abandono** (basado en valores SHAP):
1. **Edad** - Clientes mayores (>50) tienen tasas de abandono más altas
2. **Número de Productos** - Clientes con solo 1 producto tienen más probabilidad de abandonar
3. **Estado de Miembro Activo** - Miembros inactivos tienen tasa de abandono 3x mayor
4. **Geografía** - Alemania tiene la tasa de abandono más alta (32%), Francia la más baja (16%)
5. **Saldo** - Saldos muy bajos (<10K) o muy altos (>150K) se correlacionan con abandono
6. **Género** - Clientes femeninas ligeramente más propensas a abandonar (22% vs 16%)

### Segmentos de Clientes

**Perfil de Alto Riesgo**:
- Edad: 45-60 años
- Antigüedad: 0-2 años (clientes nuevos)
- Productos: Solo 1 producto
- Estado activo: Inactivo
- Saldo: Extremos (<10K o >150K)

**Perfil de Bajo Riesgo**:
- Edad: 30-40 años
- Antigüedad: 3+ años
- Productos: 2-3 productos
- Estado activo: Activo
- Saldo: Rango 50K-100K

## 🎯 Impacto de Negocio

### Propuesta de Valor

1. **Protección de Ingresos**: Retener 25-35% de clientes en riesgo
2. **Eficiencia de Costos**: 5x más barato retener que adquirir nuevos clientes
3. **Campañas Dirigidas**: Enfocar recursos en clientes de alto valor y alto riesgo
4. **Satisfacción del Cliente**: El compromiso proactivo mejora la experiencia del cliente

### Marco de Estrategia de Retención

**Riesgo Bajo** (Probabilidad < 30%):
- Servicio al cliente estándar
- Encuestas de satisfacción trimestrales
- Programa de recompensas de lealtad

**Riesgo Medio** (Probabilidad 30-60%):
- Comunicación personalizada
- Ofertas especiales en productos adicionales
- Revisión con gerente de cuenta

**Riesgo Alto** (Probabilidad > 60%):
- Intervención inmediata del equipo de retención
- Ofertas de retención personalizadas (exención de comisiones, bonos)
- Alcance a nivel ejecutivo
- Descuentos en paquetes de productos

### Análisis de ROI

**Escenario**: Banco con 100,000 clientes, tasa de abandono del 20%, CLV promedio $2,500

- **Sin Modelo**: 20,000 abandonos × $2,500 = **Pérdida anual de $50M**
- **Con Modelo**: 
  - Identificar 17,000 abandonos (85% recall)
  - Retener 30% con campañas dirigidas = 5,100 clientes
  - Ingresos salvados: 5,100 × $2,500 = **$12.75M**
  - Costo de campaña: $100 por cliente × 17,000 = $1.7M
  - **Beneficio Neto: $11.05M anualmente**
  - **ROI: 650%**

## 🛠️ Mejoras Futuras

### Corto Plazo
- [ ] Agregar gráficos de fuerza SHAP para predicciones individuales
- [ ] Implementar marco de pruebas A/B para estrategias de retención
- [ ] Crear alertas automáticas de email/SMS para clientes de alto riesgo
- [ ] Construir dashboard de seguimiento de campañas de retención
- [ ] Agregar predicciones de valor del cliente (CLV) junto con abandono

### Largo Plazo
- [ ] Modelo de deep learning (Redes Neuronales) para mejorar precisión
- [ ] API de predicción en tiempo real (FastAPI/Flask)
- [ ] Integración con sistemas CRM (Salesforce, HubSpot)
- [ ] Procesamiento de Lenguaje Natural para análisis de feedback de clientes
- [ ] Análisis de supervivencia para predicciones de tiempo hasta abandono
- [ ] Seguimiento de comportamiento de cliente multicanal (web, móvil, sucursal)

## 📚 Referencias

1. **Dataset**: Bank Customer Churn Dataset, Kaggle (2023)

2. **Investigación**: Lemmens, A., & Croux, C. (2006). "Bagging and boosting classification trees to predict churn". Journal of Marketing Research.

3. **Libro**: Neslin, S., et al. (2006). "Defection Detection: Measuring and Understanding the Predictive Accuracy of Customer Churn Models". Journal of Marketing Research.

4. **Reporte Industrial**: Bain & Company (2024). "Customer Retention Statistics and Economics".

## 👤 Autor

**Franklin Ramos**
- 📧 Email: franklin.ram.riv@gmail.com
- 💼 LinkedIn: [linkedin.com/in/franklin-ramos-riveros-62b70083](https://www.linkedin.com/in/franklin-ramos-riveros-62b70083/)
- 💼 GitHub: [github.com/frankliramos](https://github.com/frankliramos)
- Portafolio: [GitHub Portfolio](https://github.com/frankliramos/Proyectos-portafolio)

## 📄 Licencia

Este proyecto es parte de un portafolio de ciencia de datos. Ver archivo `LICENSE` para detalles.

## 🙏 Agradecimientos

- Expertos del dominio de la industria bancaria por insights de estrategia de retención
- Comunidad de código abierto por excelentes bibliotecas de ML

---

**Nota**: Este es un proyecto de portafolio para propósitos educativos y de demostración. El dataset es simulado pero refleja escenarios bancarios del mundo real. Para despliegue en producción, se requerirían consideraciones adicionales de cumplimiento, privacidad y regulatorias (GDPR, regulaciones bancarias, etc.).
