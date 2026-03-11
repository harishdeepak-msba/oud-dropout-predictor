# OUD Dropout Risk Predictor — Full-Stack Web App
**Team 3 · Deep Learning Course · SAMHSA TEDS-D 2018**

## Project Structure
```
oud_app/
├── app.py              ← FastAPI backend (API + serves frontend)
├── requirements.txt    ← Python dependencies
├── xgboost_model.pkl   ← YOUR trained model goes here (see below)
├── static/
│   └── index.html      ← Frontend (clinical prediction UI)
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
cd oud_app
pip install -r requirements.txt
```

### 2. Export the trained pipeline from your notebook
The app needs the **full sklearn Pipeline** (preprocessor + XGBoost), not just the model.
Save it as `xgboost_pipeline.pkl` in the `oud_app/` folder:

```python
# In your notebook, after Section 3 has run:
import pickle
with open("xgboost_pipeline.pkl", "wb") as f:
    pickle.dump(best_pipes["XGBoost"], f)
```

> **Why the full Pipeline?** Your notebook wraps preprocessing (imputation, scaling, OHE)
> and XGBoost into a single `Pipeline(["pre", preprocessor], ["clf", XGBClassifier])`.
> Pickling the whole Pipeline means the API just calls `pipeline.predict_proba(X_raw_df)`
> with a raw DataFrame — no manual preprocessing needed.

> **No model file?** The app runs in **demo mode** using a heuristic — all UI features
> still work, predictions just come from rules instead of XGBoost.

### 3. Run the server
```bash
uvicorn app:app --reload --port 8000
```

### 4. Open the app
Visit: **http://localhost:8000**

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Frontend UI |
| GET | `/health` | API status + model loaded check |
| POST | `/predict` | Returns dropout probability + risk factors |
| GET | `/model-info` | Model metadata |

### Example `/predict` request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_category": 5,
    "gender": 1,
    "race": 4,
    "education": 3,
    "employment": 3,
    "living_arrangement": 1,
    "primary_substance": 5,
    "route_of_admin": 4,
    "frequency_of_use": 5,
    "age_first_use": 4,
    "prior_treatment_episodes": 2,
    "service_setting": 6,
    "medication_assisted": 0,
    "state_fips": 36
  }'
```

### Example response
```json
{
  "dropout_probability": 0.7312,
  "risk_level": "High Risk",
  "risk_color": "#ef4444",
  "top_factors": [
    {"factor": "Daily or near-daily substance use", "impact": "high", "direction": "increase"},
    {"factor": "Homeless / unstable housing", "impact": "high", "direction": "increase"},
    {"factor": "Currently unemployed", "impact": "medium", "direction": "increase"}
  ],
  "recommendation": "Immediate case management intervention recommended..."
}
```

---

## Feature Mapping (matches TEDS-D columns)

| Field | TEDS-D Column | Description |
|-------|--------------|-------------|
| age_category | AGECAT | 1–12 age brackets |
| gender | GENDER | 1=Male, 2=Female |
| race | RACE | 1–7 categories |
| education | EDUC | 1–5 levels |
| employment | EMPLOY | 1=FT, 2=PT, 3=Unemployed, 4=Not in LF |
| living_arrangement | LIVARAG | 1=Homeless, 2=Dependent, 3=Independent |
| primary_substance | SUB1 | 2=Alcohol, 5=Heroin, 6=Other opiates, 7=Other |
| route_of_admin | ROUTE1 | 1–5 routes |
| frequency_of_use | FREQ1 | 1–5 frequency levels |
| age_first_use | AGE1ST | 1–7 age brackets |
| prior_treatment_episodes | NOPRIOR | 0–5 prior episodes |
| service_setting | SERVICES | 1–7 settings |
| medication_assisted | METHUSE | 0=None, 1–4 MAT types |
| state_fips | STFIPS | State FIPS code |

---

## Customization Tips

- **Different feature order?** Update `build_feature_vector()` in `app.py` to match exactly how your notebook preprocessed data before fitting.
- **Scaling/encoding?** If your pipeline included a scaler or encoder, wrap the model in a `Pipeline` before pickling, or apply transforms in `build_feature_vector()`.
- **CORS for deployment**: The app allows all origins by default. Restrict this for production.
