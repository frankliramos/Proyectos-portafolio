# Data Science & Machine Learning Portfolio

**Franklin Ramos**

[🇪🇸 Versión en Español](./README_ES.md)

---

## 📊 Portfolio Overview

Welcome to my Data Science and Machine Learning portfolio. This repository showcases professional end-to-end projects demonstrating expertise in predictive modeling, deep learning, time series forecasting, and production-ready ML systems.

### 🖥️ Interactive Dashboards

All projects include **interactive Streamlit dashboards** for real-time visualization and exploration:

- **Proyecto 1**: Engine health monitoring with RUL predictions
- **Proyecto 2**: Sales forecasting with inventory recommendations
- **Proyecto 3**: Customer churn risk assessment and retention strategies

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

### Proyecto 4: Coming Soon
**Status**: 🔜 In Planning

[📂 View Project →](./Proyecto%204)

---

## 🛠️ Technical Skills Demonstrated

### Machine Learning & Deep Learning
- **Time Series Forecasting** - LSTM, XGBoost, seasonal patterns
- **Classification** - Ensemble methods, imbalanced data handling (SMOTE)
- **Feature Engineering** - Sensor data, retail metrics, customer behavior, domain knowledge
- **Model Optimization** - Hyperparameter tuning, cross-validation
- **Model Evaluation** - MAE, RMSE, R², F1-Score, ROC-AUC, business metrics
- **Model Interpretability** - SHAP values, feature importance

### Software Engineering
- **Production Code** - Modular architecture, error handling, logging
- **Dashboard Development** - Interactive Streamlit applications
- **Data Pipelines** - ETL processes, data validation
- **Testing** - Unit tests, integration tests, data quality checks
- **Documentation** - Technical docs, user guides, model cards

### Tools & Technologies
- **Languages**: Python 3.12+
- **ML Frameworks**: PyTorch, scikit-learn, XGBoost, imbalanced-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly, Streamlit
- **Interpretability**: SHAP
- **Development**: Git, Docker, Jupyter

---

## 📊 Portfolio Metrics

| Metric | Value |
|--------|-------|
| **Total Projects** | 4 (3 complete, 1 planned) |
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
├── .gitignore                               # Global gitignore
│
├── Proyecto 1/                              # Turbofan Predictive Maintenance
│   └── turbofan-predictive-maintenance/
│       ├── app.py                           # Interactive dashboard
│       ├── README.md                        # Project documentation
│       ├── data/                            # NASA CMAPSS dataset
│       ├── models/                          # Trained models
│       ├── notebooks/                       # Jupyter analysis
│       ├── src/                             # Source code
│       └── results/                         # Model evaluation
│
├── Proyecto 2/                              # Sales Forecasting (Coming Soon)
│   ├── dashboard/                           # Streamlit app
│   ├── data/                                # Retail data
│   ├── models/                              # XGBoost models
│   ├── notebooks/                           # EDA and modeling
│   └── src/                                 # Source code
│
├── Proyecto 3/                              # Coming Soon
│   └── README.md
│
└── Proyecto 4/                              # Coming Soon
    └── README.md
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

**Last Updated**: February 2026
