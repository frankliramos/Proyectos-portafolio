# 🚀 Quick Start Guide

Get the Turbofan RUL Prediction dashboard running in minutes!

## Prerequisites

- Python 3.12+ installed
- Git installed
- 2GB free disk space

## Option 1: Local Installation (Recommended for Development)

### 1. Clone the Repository

```bash
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd Proyectos-portafolio/turbofan-predictive-maintenance
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## Option 2: Docker (Recommended for Production)

### Prerequisites
- Docker installed
- Docker Compose installed (optional but recommended)

### Using Docker Compose (Easiest)

```bash
# Clone and navigate to project
git clone https://github.com/frankliramos/Proyectos-portafolio.git
cd Proyectos-portafolio/turbofan-predictive-maintenance

# Start the dashboard
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the dashboard
docker-compose down
```

Access at: `http://localhost:8501`

### Using Docker Directly

```bash
# Build the image
docker build -t turbofan-rul .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  turbofan-rul
```

## Option 3: Cloud Deployment

### Deploy to Streamlit Cloud (Free)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your fork and choose `turbofan-predictive-maintenance/app.py`
6. Deploy!

### Deploy to Heroku

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main
```

## Verification

After starting the dashboard, you should see:

1. ✈️ **Title**: "Turbofan Engine Health Monitor"
2. **4 KPI Cards**: Motor ID, Current Cycles, Predicted RUL, Asset State
3. **Distribution Chart**: RUL distribution for all engines
4. **Sensor Evolution**: Interactive sensor plots
5. **Data Tables**: Raw data and predictions summary

## Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: "Model file not found"

**Solution:**
Ensure you're in the correct directory:
```bash
cd turbofan-predictive-maintenance
ls models/  # Should show lstm_model_v1.pth, scaler_v1.pkl, etc.
```

### Issue: "Port 8501 already in use"

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Issue: Dashboard loads slowly

**Solution:**
- First load computes predictions for all engines (cached after)
- Click "🔄 Recalcular RUL" only when necessary
- Reduce cycle range for faster visualization

## Next Steps

1. **Explore the Dashboard**: Try different engines and thresholds
2. **Read the Documentation**: Check [README.md](README.md) for details
3. **Review the Model**: See [MODEL_CARD.md](MODEL_CARD.md) for architecture
4. **Explore Notebooks**: Check `notebooks/` for analysis
5. **Customize**: Modify thresholds and sensors in the sidebar

## Features to Try

- 🎯 **Select Different Engines**: Use sidebar dropdown
- 📊 **Adjust Thresholds**: Change critical/warning levels
- 📈 **Compare Sensors**: Select multiple sensors to plot
- 📥 **Export Data**: Enable export and download CSV
- 🔄 **Recalculate**: Refresh predictions for all engines
- 🔍 **Filter States**: Show only critical or warning engines

## Performance Tips

- **First Load**: ~30 seconds to compute all predictions (cached)
- **Subsequent Loads**: < 5 seconds (uses cache)
- **Memory Usage**: ~500MB RAM
- **Disk Space**: ~300MB total

## Getting Help

- 📖 **Documentation**: [README.md](README.md)
- 🐛 **Report Issues**: [GitHub Issues](https://github.com/frankliramos/Proyectos-portafolio/issues)
- 💬 **Questions**: Check [CONTRIBUTING.md](CONTRIBUTING.md)

## Sample Workflow

1. **Start Dashboard**: `streamlit run app.py`
2. **Select Engine**: Use sidebar to pick a motor ID
3. **Review Health**: Check RUL prediction and state (🟢🟡🔴)
4. **Analyze Trends**: Plot multiple sensors over time
5. **Export Results**: Download predictions for offline analysis
6. **Adjust Thresholds**: Customize for your use case

---

**Ready to go!** 🎉 The dashboard should now be running and you can start exploring predictive maintenance insights.

Need more details? See the [full README](README.md) for comprehensive documentation.
