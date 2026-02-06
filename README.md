# 📈 Sales Forecasting - Enterprise ML System

**Production-Ready ML Platform** with 10 Advanced Features

## 🎯 ENTERPRISE FEATURES (All Implemented)

1. **Autonomous ML System** - Auto-retraining scheduler with best model auto-selection
2. **MLOps Layer** - Model registry, experiment tracking, audit logs
3. **Real-Time Forecasting** - Streaming data with anomaly detection & alerts
4. **Advanced XAI** - Counterfactual explanations + business context
5. **Agentic AI** - Multi-step reasoning with tool execution
6. **Business Decision Engine** - Inventory, pricing, expansion optimization
7. **Security & Enterprise** - RBAC, audit trails, model rollback
8. **Multi-Tenant Mode** - Complete data isolation + white-label
9. **Scalability & Architecture** - Designed for 10x growth
10. **Documentation** - Interview-ready system design

**See [README_ENTERPRISE.md](README_ENTERPRISE.md) for complete feature details.**

---

## A comprehensive machine learning pipeline for sales forecasting with data processing, feature engineering, model training, and interactive dashboard.

## 📁 Project Structure

```
sales-forecasting/
│
├── data/
│   ├── raw/                         # Original datasets (8 CSV files)
│   └── processed/                   # Master dataset (final_dataset.csv)
│
├── notebooks/                       # Jupyter notebooks (development)
│   ├── 01_build_final_dataset.ipynb # Merge all raw data
│   ├── 02_eda.ipynb                 # Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb # Feature creation
│   ├── 04_model_training.ipynb      # ML/DL models
│   ├── 05_model_evaluation.ipynb    # Performance metrics
│   └── 06_xai.ipynb                 # Explainability (SHAP)
│
├── src/                             # Reusable Python modules
│   ├── data_pipeline.py             # Data merging & loading
│   ├── features.py                  # Feature engineering
│   ├── train.py                     # Model training
│   ├── evaluate.py                  # Model evaluation
│   └── utils.py                     # Utility functions
│
├── models/                          # Saved trained models
│   └── best_model.pkl
│
├── app/                             # Streamlit dashboard
│   └── app.py                       # Interactive app
│
├── reports/                         # Output & insights
│   ├── eda_plots.png
│   ├── forecast_results.png
│   └── business_insights.md
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build Master Dataset
- Run Notebook: `01_build_final_dataset.ipynb`
- Or use: `python src/data_pipeline.py`

### 4. Exploratory Analysis
- Run Notebook: `02_eda.ipynb`

### 5. Feature Engineering
- Run Notebook: `03_feature_engineering.ipynb`

### 6. Train Models
- Run Notebook: `04_model_training.ipynb`
- Or use: `python src/train.py`

### 7. Evaluate Models
- Run Notebook: `05_model_evaluation.ipynb`

### 8. Explainability (SHAP)
- Run Notebook: `06_xai.ipynb`

### 9. Run Dashboard
```bash
streamlit run app/app.py
```

## 📊 Raw Datasets

The `data/raw/` folder contains 8 CSV files:
- `sales.csv` - Sales transactions
- `price_history.csv` - Historical pricing
- `discounts_history.csv` - Discount information
- `catalog.csv` - Product catalog
- `stores.csv` - Store information
- `online.csv` - Online sales data
- `markdowns.csv` - Markdown events
- `actual_matrix.csv` - Actual sales matrix

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `src/data_pipeline.py` | Load & merge all raw data |
| `src/features.py` | Create lag, rolling, temporal features |
| `src/train.py` | Train ML models (RF, GB, etc.) |
| `src/evaluate.py` | Calculate metrics (RMSE, MAE, R²) |
| `app/app.py` | Advanced dashboard with all 11 features |
| `app/dashboard_utils.py` | Advanced dashboard utilities & classes |

## 🚀 ADVANCED DASHBOARD v2.0 - 11 FEATURE SETS

### ✅ Feature 1️⃣: Navigation & UX Improvements
- 🔍 Global search (Store/Product/Category)
- ⭐ Favorites system
- 🌓 Dark/Light mode toggle
- ⏱️ Auto-refresh toggle
- 📱 Mobile responsive view
- 📑 Sidebar navigation (10 pages)

### ✅ Feature 2️⃣: Advanced Prediction Features
- 📅 Custom prediction range (7/14/30/90 days)
- 🎯 Confidence intervals (80%-99%)
- 🔮 Scenario prediction (discount/price changes)
- 📦 Bulk prediction (CSV upload)
- 📊 Confidence zone visualization

### ✅ Feature 3️⃣: Model Intelligence
- 🏆 Best model auto-selection
- ⚖️ Ensemble prediction (weighted averaging)
- 🧪 One-click retrain button
- 🧠 Model version history
- 📉 Drift detection alerts

### ✅ Feature 4️⃣: Advanced Analysis
- 📆 Seasonality analysis (monthly/weekly)
- 🗓️ Holiday vs non-holiday comparison
- 💸 Discount impact curves
- 🛒 Top & bottom selling products
- 🏪 Store-wise performance heatmap

### ✅ Feature 5️⃣: Explainable AI (XAI)
- 📌 SHAP summary plot (global)
- 📍 Force plot (single predictions)
- 📊 Feature importance per model
- ❓ "Why this prediction?" text explanations
- 🧠 Model decision breakdown (plain English)

### ✅ Feature 6️⃣: Model Comparison (Upgraded)
- 📊 Side-by-side metric table (RMSE, MAE, MAPE)
- 📈 Actual vs Predicted (per model)
- ⏱️ Training time comparison
- 💾 Model size & inference speed
- 🏅 Best model badge

### ✅ Feature 7️⃣: Data Quality & Monitoring
- 🚨 Missing value alerts
- 📉 Outlier detection
- 📊 Data distribution shift
- 🧼 Data cleaning summary
- 📋 Last data update log

### ✅ Feature 8️⃣: Business Insights Panel
- 🧠 Auto-generated insights
- 📈 Revenue growth suggestions
- 📉 Loss prevention alerts
- 📦 Inventory shortage prediction
- 💡 Key findings & recommendations

### ✅ Feature 9️⃣: Admin/Control Panel
- 👤 User roles (Admin/Analyst/Viewer)
- 🔐 Login & authentication
- 📤 Dataset upload/replace
- ⚙️ Feature toggle (enable/disable)
- 🗂️ Model management (delete/archive)

### ✅ Feature 🔟: Export & Integration
- 📥 Download (CSV/Excel)
- 📄 Auto-generate PDF report
- 🔗 REST API endpoint
- 📧 Email alerts for forecast changes
- 🔔 Slack/webhook integration

### ✅ Feature 1️⃣1️⃣: AI Assistant (SUPER ADVANCED)
- 🤖 Chat-based natural language interface
- 💬 Ask: "Why sales dropped?" or "Predict next month"
- 📝 Natural language to query data
- 📊 Auto chart generation
- 🧠 Insight explanation in plain English

**All features:** Available now at `http://localhost:8501` (10 pages)

## 📈 Workflow

```
raw data → merge → feature engineering → train models → evaluate → deploy
```

## 🎯 Models

- Random Forest Regressor
- Gradient Boosting Regressor
- LSTM (Neural Network)
- Ensemble Methods

## 📊 Evaluation Metrics

- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R²** - R-squared Score
- **MAPE** - Mean Absolute Percentage Error

## 🔍 Explainability

- SHAP (SHapley Additive exPlanations)
- Feature Importance
- Dependency Plots

## 📚 Dependencies

See `requirements.txt` for complete list:
- pandas, numpy
- scikit-learn
- matplotlib, seaborn, plotly
- streamlit
- tensorflow/keras
- shap

## 📝 Notes

- All raw data files should be in `data/raw/`
- Processed data is saved in `data/processed/`
- Models are pickled and saved in `models/`
- Use notebooks for experimentation
- Use `src/` modules for production code

## 🤝 Contributing

1. Create a new branch
2. Make changes
3. Test thoroughly
4. Submit pull request

## 📄 License

MIT License

---

**Created:** February 2026  
**Version:** 1.0
