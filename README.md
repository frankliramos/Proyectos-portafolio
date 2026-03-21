# Data Science & Machine Learning Portfolio

**Franklin Ramos**

[![CI](https://github.com/frankliramos/Proyectos-portafolio/actions/workflows/ci.yml/badge.svg)](https://github.com/frankliramos/Proyectos-portafolio/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

[🇪🇸 Versión en Español](./README_ES.md)

---

## 📊 Portfolio Overview

Welcome to my Data Science and Machine Learning portfolio. This repository showcases professional end-to-end projects demonstrating expertise in predictive modeling, deep learning, time series forecasting, and production-ready ML systems.

### 🖥️ Live Interactive Dashboards

All projects are **deployed and accessible online** — no installation required:

| # | Project | Live Demo |
|---|---------|-----------|
| 1 | 🔧 Turbofan Engine Health Monitor | [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyectos-portafolio-fvuxxicflgewt7jxjtdzd.streamlit.app/) |
| 2 | 📈 Sales Forecasting System | [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyectos-portafolio-amcgczthtr4a7s3epewp2q.streamlit.app/) |
| 3 | 💬 Financial Sentiment Analysis | [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyectos-portafolio-erzcz3etb9efmhgmw8qxep.streamlit.app/) |
| 4 | 📡 Churn Prediction - Telecom | [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyectos-portafolio-batbljpwqey6pemu5s2dr7.streamlit.app/) |

**Or run locally**:
```bash
# For any project with a dashboard
cd "[project-directory]"
pip install -r requirements.txt
streamlit run app.py
```

📖 **[Complete Dashboard Access Guide →](./DASHBOARD_ACCESS.md)**

---

## 🚀 Projects

### Proyecto 1: Turbofan Predictive Maintenance
**Status**: ✅ Complete | **Type**: Deep Learning, Time Series, Predictive Maintenance

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyectos-portafolio-fvuxxicflgewt7jxjtdzd.streamlit.app/)

Production-ready predictive maintenance system for aircraft turbofan engines using LSTM neural networks.

- **Business Impact**: Predicts engine failures 30-40 cycles in advance
- **Tech Stack**: PyTorch, LSTM, Streamlit Dashboard
- **Dataset**: NASA CMAPSS (200 engines, 33K+ cycles)
- **Performance**: MAE ~14.2 cycles, RMSE ~19.7, R² 0.78

[📂 View Project →](./turbofan-predictive-maintenance)

**Key Features**:
- Real-time RUL (Remaining Useful Life) predictions
- Interactive dashboard with 21 sensor streams
- Fleet management monitoring
- Comprehensive documentation (English & Spanish)

---

### Proyecto 2: Sales Forecasting System
**Status**: ✅ Complete | **Type**: Time Series, XGBoost, Demand Forecasting

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyectos-portafolio-amcgczthtr4a7s3epewp2q.streamlit.app/)

Advanced retail sales forecasting system for Ecuadorian stores using XGBoost with GPU acceleration.

- **Business Impact**: 83% prediction accuracy (WAPE 16.9%), 15-20% reduction in inventory costs
- **Tech Stack**: XGBoost GPU, Streamlit Dashboard, Pandas
- **Dataset**: 2.9M+ transactions, 54 stores, 33 product categories
- **Performance**: RMSLE 0.40, WAPE 16.9%

[📂 View Project →](./sales-forecasting)

**Key Features**:
- 15-day sales forecasting by store and product family
- Interactive dashboard with real-time predictions
- Inventory optimization recommendations
- External factors integration (oil prices, holidays)
- Comprehensive documentation (English & Spanish)

---

### Proyecto 3: Financial Sentiment Analysis
**Status**: ✅ Complete | **Type**: NLP, FinBERT, Financial Analytics

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyectos-portafolio-erzcz3etb9efmhgmw8qxep.streamlit.app/)

Advanced financial sentiment analysis system using a fine-tuned FinBERT model that classifies financial news and earnings text into negative, neutral, and positive sentiment.

- **Business Impact**: Automates analysis of 1,000+ news items per day; 25-35% improvement in analyst productivity
- **Tech Stack**: FinBERT (HuggingFace), scikit-learn, Streamlit Dashboard, FastAPI
- **Dataset**: Financial PhraseBank — 4,840 annotated financial sentences
- **Performance**: Accuracy 87.3%, F1-Score (Macro) 86.1%, ROC-AUC 0.94

[📂 View Project →](./financial-sentiment-analysis)

**Key Features**:
- Real-time sentiment classification (Positive / Neutral / Negative)
- FinBERT fine-tuned on financial corpus for domain-specific accuracy
- Interactive dashboard with keyword analysis and confidence scores
- Use cases: earnings call analysis, news monitoring, trading signals
- Comprehensive documentation (English & Spanish)

---

### Proyecto 4: Customer Churn Prediction - Telecom
**Status**: ✅ Complete | **Type**: Classification, Customer Analytics, Telecom

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyectos-portafolio-batbljpwqey6pemu5s2dr7.streamlit.app/)

Churn prediction system for a telecom provider using XGBoost. Identifies customers at risk, explains key churn drivers via SHAP, and estimates financial impact from retention actions.

- **Business Impact**: Proactive retention targeting; quantifies revenue at risk per customer
- **Tech Stack**: XGBoost, SHAP, Streamlit Dashboard
- **Dataset**: Telco Customer Churn dataset (contract, tenure, charges, services, demographics)
- **Performance**: Accuracy 0.93, Recall (Churn) 0.86, ROC-AUC 0.98

[📂 View Project →](./customer-churn-prediction)

**Key Features**:
- Customer churn risk scoring and ranking
- SHAP-based model interpretability (top churn drivers per customer)
- Financial impact estimation and retention prioritization
- Interactive dashboard with individual and cohort analysis
- Comprehensive documentation (English & Spanish)

---

## 🛠️ Technical Skills Demonstrated

### Machine Learning & Deep Learning
- **Time Series Forecasting** - LSTM, XGBoost, seasonal patterns
- **Classification** - Ensemble methods, imbalanced data handling (SMOTE)
- **Recommendation Systems** - Collaborative filtering, content-based filtering, hybrid models
- **Feature Engineering** - Sensor data, retail metrics, customer behavior, domain knowledge
- **Model Optimization** - Hyperparameter tuning, cross-validation
- **Model Evaluation** - MAE, RMSE, R², F1-Score, ROC-AUC, Precision@K, NDCG, business metrics
- **Model Interpretability** - SHAP values, feature importance

### Software Engineering
- **Production Code** - Modular architecture, error handling, logging
- **Dashboard Development** - Interactive Streamlit applications
- **Data Pipelines** - ETL processes, data validation
- **Testing** - Unit tests, integration tests, data quality checks
- **Documentation** - Technical docs, user guides, model cards

### Tools & Technologies
- **Languages**: Python 3.12+
- **ML Frameworks**: PyTorch, scikit-learn, XGBoost, TensorFlow, imbalanced-learn
- **Recommendation**: Implicit, Surprise, LightFM
- **NLP**: NLTK, spaCy, TF-IDF
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly, Streamlit
- **Interpretability**: SHAP
- **Development**: Git, Docker, Jupyter

---

## 📊 Portfolio Metrics

| Metric | Value |
|--------|-------|
| **Total Projects** | 4 completed |
| **Lines of Code** | 8,000+ |
| **Documentation Pages** | 20+ technical documents |
| **Technologies** | 15+ frameworks and tools |
| **Bilingual Docs** | English & Spanish |

---

## 🎯 Professional Approach

This portfolio demonstrates:

1. **Business Value** - Solving real-world problems with measurable impact
2. **Technical Excellence** - Production-ready code with best practices
3. **Communication** - Clear documentation for technical and non-technical audiences
4. **End-to-End Execution** - From problem definition to deployment
5. **Scalability** - Organized structure for multiple projects

---

## 🔍 Repository Structure

```
Proyectos-portafolio/
├── README.md                                # This file (English)
├── README_ES.md                             # Spanish version
├── LICENSE                                  # MIT License
├── .gitignore                               # Global gitignore
├── .github/workflows/ci.yml                 # CI/CD pipeline
│
├── turbofan-predictive-maintenance/         # Turbofan Predictive Maintenance
│   ├── app.py                           # Interactive dashboard (Streamlit entry point)
│   ├── README.md                        # Project documentation
│   ├── MODEL_CARD.md                    # Model specifications
│   ├── data/                            # NASA CMAPSS dataset (raw + processed)
│   ├── models/                          # Trained LSTM models
│   ├── notebooks/                       # Jupyter EDA & modeling notebooks
│   ├── src/                             # Source code modules
│   ├── results/                         # Model evaluation results
│   ├── dashboard/                       # Dashboard source (mirrored by root app.py)
│   └── Dockerfile                       # Container definition
│
├── sales-forecasting/                       # Sales Forecasting
│   ├── app.py                           # Streamlit entry point
│   ├── dashboard/app.py                 # Dashboard source
│   ├── notebooks/                       # EDA and modeling notebooks
│   ├── src/                             # Feature engineering & prediction
│   └── requirements.txt                 # Dependencies
│
├── financial-sentiment-analysis/            # Financial Sentiment Analysis (FinBERT)
│   ├── app.py                           # Interactive dashboard (Streamlit entry point)
│   ├── README.md                        # Project documentation
│   ├── data/                            # Financial PhraseBank dataset
│   ├── models/                          # Trained FinBERT models
│   ├── notebooks/                       # Analysis notebooks
│   ├── src/                             # Source code modules
│   └── results/                         # Evaluation results
│
└── customer-churn-prediction/               # Customer Churn Prediction (Telecom)
│   ├── app.py                           # Interactive dashboard (Streamlit entry point)
│   ├── README.md                        # Project documentation
│   ├── data/                            # Telecom customer dataset
│   ├── models/                          # Trained XGBoost models
│   ├── notebooks/                       # Analysis notebooks
│   ├── src/                             # Source code modules
│   └── results/                         # Evaluation results
```

---

## 🚀 Quick Start

### Running Turbofan Predictive Maintenance Dashboard

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd "Proyectos-portafolio/turbofan-predictive-maintenance"
pip install -r requirements.txt
streamlit run app.py
```

### Running Sales Forecasting Dashboard

```bash
cd "Proyectos-portafolio/sales-forecasting"
pip install -r requirements.txt
streamlit run app.py
```

### Running Financial Sentiment Analysis Dashboard

```bash
cd "Proyectos-portafolio/financial-sentiment-analysis"
pip install -r requirements.txt
streamlit run app.py
```

### Running Customer Churn Prediction Dashboard

```bash
cd "Proyectos-portafolio/customer-churn-prediction"
pip install -r requirements.txt
streamlit run app.py
```

Dashboard launches at `http://localhost:8501`

---

## 📬 Contact

**Franklin Ramos**

- 📧 Email: Franklin.ram.riv@gmail.com
- 💼 GitHub: [github.com/frankliramos](https://github.com/frankliramos/Proyectos-portafolio)
- 🔗 LinkedIn: [linkedin.com/in/franklin-ramos-riveros-62b70083](https://www.linkedin.com/in/franklin-ramos-riveros-62b70083/?locale=en_US)
- 🌐 Portfolio: This repository

*Open to Data Scientist / Machine Learning Engineer opportunities — focused on delivering measurable business impact with production-ready ML systems.*

---

## 📄 License

This project is available for educational and portfolio review purposes. See individual project directories for specific license information.

---

**Last Updated**: March 2026
