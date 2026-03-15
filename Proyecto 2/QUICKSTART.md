# ⚡ Quick Start Guide - Sales Forecasting

> Get the Sales Forecasting dashboard running in under 5 minutes.

## Prerequisites
- Python 3.8+
- pip package manager
- Git

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/frankliramos/Proyectos-portafolio.git

# 2. Navigate to the project
cd "Proyectos-portafolio/Proyecto 2"

# 3. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

## Running the Dashboard

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## What You'll See
- 📈 Sales trend visualizations with seasonal decomposition
- 🔮 Forecast predictions with confidence intervals
- 📊 Model performance comparison metrics
- 🎛️ Interactive filters for date range and product categories

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |
| Data not loading | Ensure you're in the correct directory |

## Project Structure
```
Proyecto 2/
├── dashboard/app.py      # Main Streamlit application
├── notebooks/             # EDA and modeling notebooks
├── src/                   # Feature engineering & prediction modules
└── requirements.txt       # Python dependencies
```

---
📧 Questions? See the main [README](./README.md) for full documentation.
