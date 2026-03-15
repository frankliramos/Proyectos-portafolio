# ⚡ Quick Start Guide - Customer Churn Prediction

> Get the Churn Prediction dashboard running in under 5 minutes.

## Prerequisites
- Python 3.8+
- pip package manager
- Git

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/frankliramos/Proyectos-portafolio.git

# 2. Navigate to the project
cd "Proyectos-portafolio/Proyecto 3"

# 3. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## What You'll See
- 🔍 Customer risk profiling with churn probability scores
- 📊 Feature importance analysis and model explanations
- 📈 Demographic and behavioral pattern visualizations
- 🎯 Interactive prediction tool for individual customers

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |
| Model not found | Check that `models/` directory contains trained model files |

## Project Structure
```
Proyecto 3/
├── app.py                # Main Streamlit dashboard
├── data/                 # Customer dataset (raw + processed)
├── models/               # Trained ensemble models
├── notebooks/            # EDA and modeling notebooks
├── src/                  # Source code modules
└── results/              # Evaluation metrics and plots
```

---
📧 Questions? See the main [README](./README.md) for full documentation.
