# Data Science & Machine Learning Portfolio

**Franklin Ramos**

[![CI](https://github.com/frankliramos/Proyectos-portafolio/actions/workflows/ci.yml/badge.svg)](https://github.com/frankliramos/Proyectos-portafolio/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

[🇪🇸 Versión en Español](./README_ES.md)

---

## 📊 Portfolio Overview

Welcome to my Data Science and Machine Learning portfolio. This repository showcases professional end-to-end projects demonstrating expertise in predictive modeling, deep learning, time series forecasting, and production-ready ML systems.

### 🖥️ Interactive Dashboards

All projects include **interactive Streamlit dashboards** for real-time visualization and exploration:

- **Proyecto 1**: Engine health monitoring with RUL predictions
- **Proyecto 2**: Sales forecasting with inventory recommendations
- **Proyecto 3**: Customer churn risk assessment and retention strategies
- **Proyecto 4**: Product recommendations and e-commerce analytics

**Quick Start**:
```bash
# For any project with a dashboard
cd "Proyecto X/[project-directory]"
pip install -r requirements.txt
streamlit run app.py
```

📖 **[Complete Dashboard Access Guide →](./DASHBOARD_ACCESS.md)**

---

## 🚀 Projects

### Proyecto 1: Turbofan Predictive Maintenance
**Status**: ✅ Complete | **Type**: Deep Learning, Time Series, Predictive Maintenance

Production-ready predictive maintenance system for aircraft turbofan engines using LSTM neural networks.

- **Business Impact**: Predicts engine failures 30-40 cycles in advance
- **Tech Stack**: PyTorch, LSTM, Streamlit Dashboard
- **Dataset**: NASA CMAPSS (200 engines, 33K+ cycles)
- **Performance**: MAE ~14.2 cycles, RMSE ~19.7, R² 0.78

[📂 View Project →](./Proyecto%201/turbofan-predictive-maintenance)

**Key Features**:
- Real-time RUL (Remaining Useful Life) predictions
- Interactive dashboard with 21 sensor streams
- Fleet management monitoring
- Comprehensive documentation (English & Spanish)

---

### Proyecto 2: Sales Forecasting System
**Status**: ✅ Complete | **Type**: Time Series, XGBoost, Demand Forecasting

Advanced retail sales forecasting system for Ecuadorian stores using XGBoost with GPU acceleration.

- **Business Impact**: 83% prediction accuracy (WAPE 16.9%), 15-20% reduction in inventory costs
- **Tech Stack**: XGBoost GPU, Streamlit Dashboard, Pandas
- **Dataset**: 2.9M+ transactions, 54 stores, 33 product categories
- **Performance**: RMSLE 0.40, WAPE 16.9%

[📂 View Project →](./Proyecto%202)

**Key Features**:
- 15-day sales forecasting by store and product family
- Interactive dashboard with real-time predictions
- Inventory optimization recommendations
- External factors integration (oil prices, holidays)
- Comprehensive documentation (English & Spanish)

---

### Proyecto 3: Customer Churn Prediction System
**Status**: ✅ Complete | **Type**: Classification, Customer Analytics, Banking

Advanced customer churn prediction system for banking institutions using ensemble machine learning.

- **Business Impact**: 25-35% churn reduction, $11M+ annual savings, 650% ROI
- **Tech Stack**: XGBoost, Random Forest, SMOTE, Streamlit Dashboard
- **Dataset**: 10,000 customers with demographics and banking behavior
- **Performance**: 86.5% accuracy, F1-Score 82.7%, ROC-AUC 0.91

[📂 View Project →](./Proyecto%203)

**Key Features**:
- Real-time churn risk assessment (individual and batch)
- Interactive customer segmentation dashboard
- Personalized retention strategy recommendations
- SHAP-based model interpretability
- Comprehensive documentation (English & Spanish)

---

### Proyecto 4: Product Recommendation System
**Status**: ✅ Complete | **Type**: Recommendation Systems, E-commerce, Personalization

Advanced hybrid recommendation engine combining collaborative and content-based filtering for e-commerce personalization.

- **Business Impact**: 20-30% conversion lift, 85% revenue increase per user, 2x CTR
- **Tech Stack**: Collaborative Filtering (ALS), Content-Based (TF-IDF), Hybrid Model, Streamlit
- **Dataset**: 50,000+ users, 10,000+ products, 500,000+ interactions
- **Performance**: Precision@10: 0.341, NDCG@10: 0.412, ROI: 1,567%-2,433%

[📂 View Project →](./Proyecto%204)

**Key Features**:
- Personalized product recommendations with confidence scores
- Similar product discovery engine
- Multi-algorithm support (Collaborative, Content-Based, Hybrid, Neural CF)
- Interactive dashboard with real-time recommendations
- A/B testing framework and analytics
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
├── Proyecto 1/                              # Turbofan Predictive Maintenance
│   └── turbofan-predictive-maintenance/
│       ├── app.py                           # Interactive dashboard
│       ├── README.md                        # Project documentation
│       ├── MODEL_CARD.md                    # Model specifications
│       ├── data/                            # NASA CMAPSS dataset (raw + processed)
│       ├── models/                          # Trained LSTM models
│       ├── notebooks/                       # Jupyter EDA & modeling notebooks
│       ├── src/                             # Source code modules
│       ├── results/                         # Model evaluation results
│       └── Dockerfile                       # Container definition
│
├── Proyecto 2/                              # Sales Forecasting
│   ├── dashboard/app.py                     # Streamlit app
│   ├── notebooks/                           # EDA and modeling notebooks
│   ├── src/                                 # Feature engineering & prediction
│   └── requirements.txt                     # Dependencies
│
├── Proyecto 3/                              # Customer Churn Prediction
│   ├── app.py                               # Interactive dashboard
│   ├── README.md                            # Project documentation
│   ├── data/                                # Customer dataset
│   ├── models/                              # Trained ensemble models
│   ├── notebooks/                           # Analysis notebooks
│   ├── src/                                 # Source code modules
│   └── results/                             # Evaluation results
│
└── Proyecto 4/                              # Product Recommendation System
    ├── app.py                               # Interactive dashboard
    ├── README.md                            # Project documentation
    ├── data/                                # E-commerce interaction data
    ├── models/                              # Trained recommendation models
    ├── notebooks/                           # Analysis notebooks
    ├── src/                                 # Source code modules
    └── results/                             # Evaluation results
```

---

## 🚀 Quick Start

### Running Proyecto 1 Dashboard

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd "Proyectos-portafolio/Proyecto 1/turbofan-predictive-maintenance"
pip install -r requirements.txt
streamlit run app.py
```

Dashboard launches at `http://localhost:8501`

---

## 📬 Contact

**Franklin Ramos**

- 📧 Email: Available upon request
- 💼 GitHub: [github.com/frankliramos](https://github.com/frankliramos)
- 🌐 Portfolio: This repository

---

## 📄 License

This project is available for educational and portfolio review purposes. See individual project directories for specific license information.

---

**Last Updated**: March 2026
