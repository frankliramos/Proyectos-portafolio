# ⚡ Quick Start Guide - Product Recommendation System

> Get the Recommendation System dashboard running in under 5 minutes.

## Prerequisites
- Python 3.8+
- pip package manager
- Git

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/frankliramos/Proyectos-portafolio.git

# 2. Navigate to the project
cd "Proyectos-portafolio/Proyecto 4"

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
- 🛍️ Personalized product recommendations for users
- 🔄 Comparison of collaborative, content-based, and hybrid approaches
- 📊 Recommendation quality metrics and A/B test results
- 💰 Business impact analysis (revenue uplift, engagement)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |
| Memory error | Reduce dataset size in `src/config.py` |

## Project Structure
```
Proyecto 4/
├── app.py                # Main Streamlit dashboard
├── data/                 # E-commerce interaction data
├── models/               # Trained recommendation models
├── notebooks/            # Analysis notebooks
├── src/                  # Source code (collaborative, content, hybrid)
├── results/              # Evaluation results
└── reports/              # Business analysis reports
```

---
📧 Questions? See the main [README](./README.md) for full documentation.
