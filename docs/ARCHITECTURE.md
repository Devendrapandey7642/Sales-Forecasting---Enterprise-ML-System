
# Sales Forecasting System - Complete Architecture

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                         │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Streamlit Dashboard (Multi-Tenant Web App)             │ │
│    │   - 10+ Pages (Home, Predictions, Analysis, etc)         │ │
│    │   - Real-time visualizations                             │ │
│    │   - Role-based access (RBAC)                             │ │
│    └──────────────────────────────────────────────────────────┘ │
│                           ↓                                      │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   AI Assistant Layer (Agentic)                           │ │
│    │   - NLP query parsing                                    │ │
│    │   - Multi-step reasoning                                 │ │
│    │   - Tool calling & execution                             │ │
│    └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                         │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Decision Intelligence Engine                           │ │
│    │   - Inventory Optimizer                                  │ │
│    │   - Revenue Loss Predictor                               │ │
│    │   - Profit Optimizer                                     │ │
│    │   - Store Expansion Analyzer                             │ │
│    └──────────────────────────────────────────────────────────┘ │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Advanced XAI Engine                                    │ │
│    │   - Counterfactual explanations                          │ │
│    │   - Human explanations                                   │ │
│    │   - Per-product/store analysis                           │ │
│    │   - PDF report generation                                │ │
│    └──────────────────────────────────────────────────────────┘ │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Real-Time Processing                                  │ │
│    │   - Stream data simulator                                │ │
│    │   - Batch prediction engine                              │ │
│    │   - Anomaly detection                                    │ │
│    │   - Alert system                                         │ │
│    └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ML OPERATIONS LAYER                           │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   MLOps Management                                       │ │
│    │   - Model Registry (versioning)                          │ │
│    │   - Experiment Tracking                                  │ │
│    │   - Performance Monitoring                               │ │
│    │   - Audit Logging                                        │ │
│    └──────────────────────────────────────────────────────────┘ │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Auto-Retraining Pipeline                               │ │
│    │   - Scheduler (daily/weekly/monthly)                     │ │
│    │   - Best model selection                                 │ │
│    │   - Hyperparameter tuning                                │ │
│    │   - Auto-promotion to production                         │ │
│    └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   CORE ML LAYER                                  │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Models (Ensemble)                                      │ │
│    │   - LinearRegression (baseline)                          │ │
│    │   - RandomForest (tuned)                                 │ │
│    │   - GradientBoosting (tuned)                             │ │
│    │   - Best model selection via registry                    │ │
│    └──────────────────────────────────────────────────────────┘ │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Feature Engineering                                    │ │
│    │   - Temporal features                                    │ │
│    │   - Lag features                                         │ │
│    │   - Rolling aggregates                                   │ │
│    └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Data Pipeline                                          │ │
│    │   ┌───────────────────────────────────────────────────┐ │ │
│    │   │ Raw Data (8 CSV sources)                         │ │ │
│    │   │ - sales.csv, price_history.csv                  │ │ │
│    │   │ - discounts_history.csv, markdowns.csv           │ │ │
│    │   │ - online.csv, stores.csv, catalog.csv           │ │ │
│    │   │ - actual_matrix.csv                              │ │ │
│    │   └───────────────────────────────────────────────────┘ │ │
│    │         ↓                                                 │ │
│    │   ┌───────────────────────────────────────────────────┐ │ │
│    │   │ Data Integration & Transformation                │ │ │
│    │   │ - Merge on keys (item_id, store_id, date)        │ │ │
│    │   │ - Handle missing values                           │ │ │
│    │   │ - Type conversions                                │ │ │
│    │   └───────────────────────────────────────────────────┘ │ │
│    │         ↓                                                 │ │
│    │   ┌───────────────────────────────────────────────────┐ │ │
│    │   │ Feature Engineering                              │ │ │
│    │   │ - final_dataset.csv (146,608 × 40 features)      │ │ │
│    │   │ - featured_dataset.csv (ready for modeling)      │ │ │
│    │   └───────────────────────────────────────────────────┘ │ │
│    │         ↓                                                 │ │
│    │   ┌───────────────────────────────────────────────────┐ │ │
│    │   │ Tenant-Specific Data (Multi-Tenant)              │ │ │
│    │   │ - Isolated per tenant in separate directories     │ │ │
│    │   └───────────────────────────────────────────────────┘ │ │
│    └──────────────────────────────────────────────────────────┘ │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Data Persistence                                       │ │
│    │   - CSV: Raw/processed data                              │ │
│    │   - PKL: Models, scalers                                 │ │
│    │   - JSON: Registry, experiments, configs                 │ │
│    └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│               INFRASTRUCTURE & SECURITY                          │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Security Layer                                         │ │
│    │   - Authentication (user/password)                       │ │
│    │   - Authorization (RBAC - 4 roles)                       │ │
│    │   - Session management                                  │ │
│    │   - Rate limiting (API)                                  │ │
│    │   - Audit logging (compliance)                           │ │
│    └──────────────────────────────────────────────────────────┘ │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │   Multi-Tenancy                                          │ │
│    │   - Data isolation                                       │ │
│    │   - Resource quotas                                      │ │
│    │   - White-label support                                  │ │
│    │   - Billing & usage tracking                             │ │
│    └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Data Flow Architecture

```
REQUEST INGESTION
     ↓
┌─ Input Data/Query ─┐
│  - Batch data      │
│  - Real-time stream│
│  - Query text (AI) │
└────────────────────┘
     ↓
PROCESSING
     ↓
┌─ Feature Extraction ──────────────┐
│ - Temporal features               │
│ - Lag/rolling features            │
│ - Categorical encoding            │
│ - Scaling (StandardScaler)        │
└───────────────────────────────────┘
     ↓
┌─ Model Prediction ────────────────┐
│ - Load production model            │
│ - Generate point predictions       │
│ - Calculate confidence intervals   │
│ - Generate explanations            │
└───────────────────────────────────┘
     ↓
┌─ Post-Processing ─────────────────┐
│ - Anomaly detection               │
│ - Real-time alerts                │
│ - Business logic (inventory, etc.)│
│ - XAI generation                  │
└───────────────────────────────────┘
     ↓
OUTPUT DELIVERY
     ↓
┌─ Response Formats ────────────────┐
│ - Dashboard visualization         │
│ - API JSON response               │
│ - CSV export                      │
│ - PDF report                      │
│ - Real-time alerts                │
└───────────────────────────────────┘
```

## 3. Component Interactions

### Model Deployment Pipeline
```
Train Data → Data Processing → Model Training → Model Registry
                                    ↓
                            Experiment Tracking
                                    ↓
                        Performance Evaluation
                                    ↓
                    [Compare vs Production]
                                    ↓
                        Auto-Promote Decision
                                    ↓
                    Production Model Update
```

### Real-Time Processing Pipeline
```
Streaming Data → Batch Buffering → Feature Extraction → Predictions
                                                            ↓
                                        Anomaly Detection/Alerts
                                                            ↓
                                        Dashboard Update
```

### Decision Engine Pipeline
```
ML Predictions → Business Logic (Inventory/Profit/Expansion) 
                                    ↓
                        Executive Recommendations
                                    ↓
                        Action Items for Users
```

## 4. Scaling Strategy

### Horizontal Scaling
- **Load Balancing**: Run multiple Streamlit instances behind load balancer
- **Model Serving**: Deploy models via FastAPI/Flask for horizontal scaling
- **Database**: Move from JSON to PostgreSQL for concurrent access
- **Caching**: Redis for prediction caching across instances

### Vertical Scaling
- **Batch Processing**: Increase batch sizes for bulk predictions
- **Parallelization**: GridSearchCV uses n_jobs=-1 (all cores)
- **Memory Optimization**: Use pandas chunking for large datasets

### Cache Strategy
```
Level 1: In-Memory Cache (Python objects)
  - Production model (pickle): ~50MB
  - Scaler object: ~1MB
  - Feature importance: ~100KB
  
Level 2: Redis Cache
  - Predictions (TTL: 1 hour): ~1GB
  - Model stats (TTL: 24 hours): ~100MB
  - User sessions (TTL: 30 min)
  
Level 3: File Cache (SSD)
  - Datasets (raw/processed): ~5GB
  - Model versions: ~50MB each
  - Experiment results: ~100MB
```

### Performance Targets
- **Prediction Latency**: <100ms for single prediction
- **Batch Processing**: 10,000 predictions/sec
- **Dashboard Load**: <2 seconds
- **Real-time Alerts**: <5 seconds from anomaly to notification

## 5. Technology Stack

```
Frontend:
  - Streamlit (Python web framework)
  - Plotly (Interactive visualizations)
  - CSS (Custom styling)

Backend:
  - Python 3.14
  - Pandas/NumPy (Data processing)
  - Scikit-learn (ML models)
  - Pickle (Model serialization)

Data Storage:
  - CSV (Raw/processed data)
  - JSON (Config/metadata)
  - PKL (Binary models/scalers)

Infrastructure:
  - Virtual environment (.venv)
  - Local deployment (can scale to cloud)
  - Multi-tenant ready for SaaS
```

## 6. Security & Compliance

### Authentication & Authorization
- User credentials hashed (SHA-256)
- Role-based access control (4 roles)
- Session management with timeout
- API rate limiting (1000 req/min default)

### Audit & Compliance
- Complete audit logging of all actions
- Timestamp tracking on all operations
- User attribution for compliance
- Exportable audit trails

### Data Privacy
- Data isolation per tenant
- No cross-tenant data leakage
- Secure model versioning
- Encrypted sensitive fields (ready)

## 7. Enterprise Features Implemented

[OK] MLOps Layer (Model Registry, Experiment Tracking)
[OK] Auto-Retraining (Weekly/Monthly Scheduler)
[OK] Real-Time Processing (Streaming + Alerts)
[OK] Advanced XAI (Counterfactuals, Explanations, PDF Reports)
[OK] Business Decision Engine (Inventory, Profit, Expansion)
[OK] Agentic AI (Tool Calling, Multi-step Reasoning)
[OK] Security Layer (RBAC, Audit Logs, Rollback)
[OK] Multi-Tenant Support (Data Isolation, White-Label)

## 8. Deployment Architecture

### Development
```
Local Machine
  ├── Python venv
  ├── Data/ (Local CSV files)
  ├── Models/ (Local PKL files)
  └── Streamlit dashboard
```

### Production (Recommended)
```
AWS/GCP/Azure
  ├── Load Balancer
  ├── Multiple Streamlit Instances
  ├── FastAPI Model Server (Inference)
  ├── PostgreSQL Database
  ├── Redis Cache
  ├── S3/Cloud Storage (Data/Models)
  ├── Scheduled Jobs (Auto-retraining)
  └── Monitoring & Logging (CloudWatch/Stackdriver)
```

### Multi-Tenant SaaS
```
API Gateway (FastAPI)
  ├── Auth/Rate Limiting middleware
  ├── Model Serving Service (multi-tenant)
  ├── Data Processing Service (isolated)
  ├── Notification Service (alerts)
  └── Admin Dashboard (tenant management)
```

## 9. Interview Highlights

**"I designed an enterprise ML system with the following components:"**

1. **Data Pipeline**: Integrated 8+ data sources into unified feature set
2. **Model Versioning**: MLOps layer with automatic model registry and promotion
3. **Auto-Retraining**: Scheduled pipeline with hyperparameter tuning
4. **Real-Time Processing**: Stream processing with anomaly detection
5. **XAI**: Counterfactual explanations + per-product insights
6. **Agentic AI**: Multi-step reasoning with tool execution
7. **Business Intelligence**: Actionable recommendations (inventory, pricing, expansion)
8. **Security**: RBAC, audit logs, rate limiting
9. **Multi-Tenancy**: Complete data isolation with white-label support
10. **Scalability**: Horizontal/vertical scaling ready for cloud deployment

**Systems Design Questions I Can Answer:**
- "How would you scale to 1M predictions/day?" (Caching, async jobs, batch processing)
- "How do you ensure data isolation in multi-tenant?" (Separate tenant IDs, path-based isolation)
- "What about model monitoring in production?" (Performance decay detection, drift detection)
- "How do you handle model rollback?" (Model registry with versioning)
- "What's your retraining strategy?" (Scheduled jobs + performance monitoring triggers)

