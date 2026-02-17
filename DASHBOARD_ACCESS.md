# 🖥️ Dashboard Access Guide

## How to View Project Dashboards

This portfolio includes interactive **Streamlit dashboards** for all four projects. Follow these simple steps to view them on your local machine.

---

## 📋 Prerequisites

- **Python 3.8+** installed on your computer
- **Git** for cloning the repository
- **Terminal/Command Prompt** access

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Clone the Repository

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd Proyectos-portafolio
```

### Step 2: Choose a Project

#### Option A: Proyecto 1 - Predictive Maintenance Dashboard

```bash
cd "Proyecto 1/turbofan-predictive-maintenance"
pip install -r requirements.txt
streamlit run app.py
```

**What you'll see**: 
- Engine health monitoring
- Remaining Useful Life (RUL) predictions
- Sensor data visualization
- Fleet-wide analytics

**Dashboard opens at**: `http://localhost:8501`

#### Option B: Proyecto 2 - Sales Forecasting Dashboard

```bash
cd "Proyecto 2/dashboard"
pip install -r ../requirements.txt
streamlit run app.py
```

**What you'll see**:
- Sales predictions by store and product
- Inventory recommendations
- Forecast accuracy metrics
- Demand drivers analysis

**Dashboard opens at**: `http://localhost:8501`

**Note**: Proyecto 2 requires `data_forecast.csv` in the dashboard directory. If this file is missing, please contact me for the dataset.

#### Option C: Proyecto 3 - Customer Churn Prediction Dashboard

```bash
cd "Proyecto 3"
pip install -r requirements.txt
streamlit run app.py
```

**What you'll see**:
- Individual customer churn risk assessment
- Customer segmentation by risk level
- Feature importance visualization
- Retention strategy recommendations
- Real-time predictions with demo data

**Dashboard opens at**: `http://localhost:8501`

**Note**: Proyecto 3 includes sample data for demonstration. The dashboard is fully functional without additional data files.

#### Option D: Proyecto 4 - Product Recommendation Dashboard

```bash
cd "Proyecto 4"
pip install -r requirements.txt
streamlit run app.py
```

**What you'll see**:
- Personalized product recommendations
- Similar product discovery
- User behavior analytics
- A/B testing results visualization
- Real-time recommendation engine with demo data

**Dashboard opens at**: `http://localhost:8501`

**Note**: Proyecto 4 includes sample data for demonstration. The dashboard generates interactive recommendations without requiring external datasets.

---

## 🔧 Troubleshooting

### Issue: "Command 'streamlit' not found"

**Solution**: Make sure you've installed the requirements:
```bash
pip install -r requirements.txt
```

### Issue: "Module not found" errors

**Solution**: Install project dependencies:
```bash
# For Proyecto 1
cd "Proyecto 1/turbofan-predictive-maintenance"
pip install -r requirements.txt

# For Proyecto 2
cd "Proyecto 2"
pip install -r requirements.txt

# For Proyecto 3
cd "Proyecto 3"
pip install -r requirements.txt
```

### Issue: Dashboard won't open

**Solution**: Manually open in your browser:
```
http://localhost:8501
```

### Issue: Port already in use

**Solution**: Stop other Streamlit instances or specify a different port:
```bash
streamlit run app.py --server.port 8502
```

---

## 📱 Dashboard Features

### Proyecto 1: Predictive Maintenance
- ✅ **100% Functional** - All data and models included
- 🔧 Select individual engines (1-100)
- 📊 View 21 sensor measurements
- ⚡ Real-time RUL predictions
- 🎯 Health status indicators (Healthy/Warning/Critical)

### Proyecto 2: Sales Forecasting
- ✅ **Functional** (with data file)
- 🏬 Select store (1-54)
- 📦 Choose product category (33 families)
- 📈 15-day sales forecast
- 💰 Inventory optimization suggestions

### Proyecto 3: Customer Churn Prediction
- ✅ **100% Functional** - Demo data included
- 👤 Individual customer risk assessment
- 📊 Customer segmentation dashboard
- 🎯 Risk classification (Low/Medium/High)
- 💡 Personalized retention strategies
- 📈 Feature importance and analytics

---

## 💼 For Clients & Recruiters

All three dashboards are **production-ready prototypes** demonstrating:

1. **Real-time Analytics**: Interactive data exploration
2. **Business Insights**: Actionable predictions and recommendations
3. **User Experience**: Clean, professional interfaces
4. **Technical Skills**: Python, ML models, data visualization, web apps

**Want to see the dashboards without local setup?** Contact Franklin for:
- Hosted demo links
- Video walkthroughs
- Live demonstration sessions

---

## 📧 Support

If you encounter any issues accessing the dashboards:

**Franklin Ramos**
- GitHub: [@frankliramos](https://github.com/frankliramos)
- Repository: [Proyectos-portafolio](https://github.com/frankliramos/Proyectos-portafolio)

---

## 🎥 Video Tutorials (Coming Soon)

- [ ] Proyecto 1 Dashboard Walkthrough
- [ ] Proyecto 2 Dashboard Walkthrough
- [ ] Proyecto 3 Dashboard Walkthrough
- [ ] Installation Guide for Windows
- [ ] Installation Guide for Mac/Linux

---

**Last Updated**: February 2026
