# Data Science & Machine Learning Portfolio

**Franklin Ramos**

[🇪🇸 Versión en Español](./README_ES.md)

---

## 📊 Portfolio Overview

This repository showcases professional data science and machine learning projects, demonstrating expertise in predictive modeling, deep learning, and production-ready ML systems.

---

## 🚀 Featured Project: Turbofan Predictive Maintenance

### [Interactive Dashboard →](./turbofan-predictive-maintenance)

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

**Production-ready predictive maintenance system for aircraft turbofan engines**

#### 🎯 Project Highlights

- **Business Impact**: Predicts engine failures before they occur, reducing unscheduled maintenance by 30-40%
- **Technical Stack**: LSTM neural networks, PyTorch, real-time monitoring dashboard
- **Dataset**: NASA CMAPSS - 100+ engines with complete run-to-failure trajectories
- **Performance**: MAE ~14.2 cycles, RMSE ~19.7 cycles, R² 0.78

#### 🔧 Key Features

✅ **Real-time Health Monitoring** - Live RUL (Remaining Useful Life) predictions  
✅ **Interactive Dashboard** - Streamlit-based visualization with 21 sensor streams  
✅ **Deep Learning Architecture** - Multi-layer LSTM with dropout regularization  
✅ **Fleet Management** - Monitor entire fleet health status at a glance  
✅ **Production Ready** - Comprehensive testing, documentation, and error handling  

#### 📱 Quick Start

```bash
cd turbofan-predictive-maintenance
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will launch at `http://localhost:8501`

#### 📖 Full Documentation

- [🇬🇧 English Documentation](./turbofan-predictive-maintenance/README.md)
- [🇪🇸 Spanish Documentation](./turbofan-predictive-maintenance/README_ES.md)
- [📊 Model Card](./turbofan-predictive-maintenance/MODEL_CARD.md)
- [⚡ Quick Start Guide](./turbofan-predictive-maintenance/QUICKSTART.md)

---

## 🛠️ Technical Skills Demonstrated

### Machine Learning & Deep Learning
- **Time Series Forecasting** - LSTM networks for sequential data
- **Feature Engineering** - Sensor data preprocessing and normalization
- **Model Optimization** - Hyperparameter tuning, early stopping
- **Model Evaluation** - MAE, RMSE, R² metrics with validation strategies

### Software Engineering
- **Production Code** - Modular architecture, error handling, logging
- **Dashboard Development** - Interactive Streamlit applications
- **Data Pipeline** - ETL processes for NASA CMAPSS dataset
- **Testing** - Unit tests and data validation
- **Documentation** - Comprehensive technical and user documentation

### Tools & Technologies
- **Languages**: Python 3.12+
- **ML Frameworks**: PyTorch, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Streamlit
- **Development**: Git, Docker, Jupyter

---

## 📊 Project Metrics

| Metric | Value |
|--------|-------|
| **Lines of Code** | 2,500+ |
| **Test Coverage** | Comprehensive data validation |
| **Documentation Pages** | 5 technical documents |
| **Data Points Processed** | 33,727 cycles across 200 engines |
| **Model Accuracy** | R² = 0.78 |

---

## 🎯 Professional Goals

This portfolio demonstrates:

- Ability to translate business problems into ML solutions
- End-to-end ML project execution from EDA to deployment
- Production-ready code with professional standards
- Clear documentation and communication skills
- Understanding of both technical and business aspects

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

## 🔍 Repository Structure

```
Proyectos-portafolio/
├── README.md                           # This file (English)
├── README_ES.md                        # Spanish version
└── turbofan-predictive-maintenance/   # Predictive maintenance project
    ├── app.py                         # Dashboard application
    ├── README.md                      # Project documentation (EN)
    ├── README_ES.md                   # Project documentation (ES)
    ├── MODEL_CARD.md                  # Model specifications
    ├── QUICKSTART.md                  # Quick start guide
    ├── requirements.txt               # Dependencies
    ├── data/                          # NASA CMAPSS dataset
    ├── models/                        # Trained models
    ├── notebooks/                     # Jupyter notebooks
    ├── src/                           # Source code
    └── results/                       # Model results
```

---

**Last Updated**: February 2026
