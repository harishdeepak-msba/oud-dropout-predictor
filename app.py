"""
OUD Treatment Dropout Prediction — FastAPI Backend
====================================================
Verified against Problem2_OUD_Team3 notebook outputs:

Dataset : SAMHSA TEDS-D 2018  (95,432 rows, 28 cols incl. DROPOUT)
Target  : DROPOUT  (1 = early dropout, 0 = completed / transferred)

The 27 training columns (df_model_base.drop(columns=["DROPOUT"])):
  alcohol_reported_at_admission                                              int64
  arrests_in_past_30_days_prior_to_admission                                object
  education                                                                  object
  employment_status_at_admission                                             object
  ethnicity                                                                  object
  gender                                                                     object
  inhalants_reported_at_admission                                            int64
  living_arrangements_at_admission                                           object
  marijuana_or_hashish_reported_at_admission                                 int64
  marital_status                                                             object
  previous_substance_use_treatment_episodes                                  object
  number_of_substances_reported_at_admission                                 int64
  referral_source                                                            object
  co_occurring_mental_and_substance_use_disorders                            object
  race                                                                       object
  census_region                                                              int64
  type_of_treatment_service_or_setting_at_admission                         int64
  veteran_status                                                             object
  census_state_fips_code                                                     int64
  maximum_frequency_of_use_of_the_three_drugs_...                           object
  are_any_of_the_drugs_used_by_injection                                    int64
  use_of_stimulants                                                          int64
  use_of_benzodiazepines_or_tranquilizers                                   int64
  use_of_other_sedatives_or_hypnotics_or_barbiturates                       int64
  use_of_hallucinogens_or_pcp_or_other_drug                                 int64
  ever_used_heroin                                                           int64
  age_category                                                               int64

Preprocessing (Pipeline step "pre"):
  ColumnTransformer:
    numeric  → SimpleImputer(median) + StandardScaler
    object   → SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")

Model saved as: xgboost_pipeline.pkl  (full sklearn Pipeline, NOT just clf)

Export from notebook:
    import pickle
    with open("xgboost_pipeline.pkl", "wb") as f:
        pickle.dump(best_pipes["XGBoost"], f)

Performance (from notebook Cell 14 / 17):
  XGBoost  ROC-AUC=0.7160  F1=0.7263  Recall=0.8159  Precision=0.6544
"""

import os, pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="OUD Dropout Prediction API", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── TEDS-D code → exact notebook column name (verified from Cell 2 output) ────
# Key: original TEDS-D column code
# Value: snake_case name produced by clean(meta["name"]) in the notebook
CODE_TO_COL = {
    # Top-10 EDA features (Cramér's V, Section 2b)
    "PSOURCE":   "referral_source",
    "SERVICES":  "type_of_treatment_service_or_setting_at_admission",
    "NUMSUBS":   "number_of_substances_reported_at_admission",
    "PSYPROB":   "co_occurring_mental_and_substance_use_disorders",
    "NOPRIOR":   "previous_substance_use_treatment_episodes",
    "LIVARAG":   "living_arrangements_at_admission",
    "EMPLOY":    "employment_status_at_admission",
    "ARRESTS":   "arrests_in_past_30_days_prior_to_admission",
    "NEEDLEUSE": "are_any_of_the_drugs_used_by_injection",
    "AGECAT":    "age_category",
    # Remaining training columns
    "GENDER":    "gender",
    "RACE":      "race",
    "EDUC":      "education",
    "ETHNIC":    "ethnicity",
    "MARSTAT":   "marital_status",
    "VETERAN":   "veteran_status",
    "METHUSE":   "maximum_frequency_of_use_of_the_three_drugs_heroin_non_prescription_methadone_other_opiates_and_synthetics",
    "STFIPS":    "census_state_fips_code",
    "CBSA":      "census_region",
    "HERFLG":    "ever_used_heroin",
    "ALCFLG":    "alcohol_reported_at_admission",
    "MARFLG":    "marijuana_or_hashish_reported_at_admission",
    "INHFLG":    "inhalants_reported_at_admission",
    "IDU":       "are_any_of_the_drugs_used_by_injection",   # same as NEEDLEUSE
    "STIMFLG":   "use_of_stimulants",
    "BENZFLG":   "use_of_benzodiazepines_or_tranquilizers",
    "SEDHPFLG":  "use_of_other_sedatives_or_hypnotics_or_barbiturates",
    "HALLFLG":   "use_of_hallucinogens_or_pcp_or_other_drug",
}

# All 27 training column names in the order X was built
ALL_TRAINING_COLS = [
    "alcohol_reported_at_admission",
    "arrests_in_past_30_days_prior_to_admission",
    "education",
    "employment_status_at_admission",
    "ethnicity",
    "gender",
    "inhalants_reported_at_admission",
    "living_arrangements_at_admission",
    "marijuana_or_hashish_reported_at_admission",
    "marital_status",
    "previous_substance_use_treatment_episodes",
    "number_of_substances_reported_at_admission",
    "referral_source",
    "co_occurring_mental_and_substance_use_disorders",
    "race",
    "census_region",
    "type_of_treatment_service_or_setting_at_admission",
    "veteran_status",
    "census_state_fips_code",
    "maximum_frequency_of_use_of_the_three_drugs_heroin_non_prescription_methadone_other_opiates_and_synthetics",
    "are_any_of_the_drugs_used_by_injection",
    "use_of_stimulants",
    "use_of_benzodiazepines_or_tranquilizers",
    "use_of_other_sedatives_or_hypnotics_or_barbiturates",
    "use_of_hallucinogens_or_pcp_or_other_drug",
    "ever_used_heroin",
    "age_category",
]

# ── Model loading ──────────────────────────────────────────────────────────────
PIPELINE = None

def load_model():
    global PIPELINE
    try:
        path = os.environ.get("MODEL_PATH", "xgboost_pipeline.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                PIPELINE = pickle.load(f)
            print(f"✅ Pipeline loaded from {path}")
        else:
            print("⚠️  xgboost_pipeline.pkl not found — running in DEMO mode")
    except Exception as e:
        print(f"⚠️  Model load failed: {e} — running in DEMO mode")

load_model()

# ── Pydantic input schema ──────────────────────────────────────────────────────
# Required: the 10 EDA features. Optional: everything else (imputed if missing).
class PatientInput(BaseModel):
    # Required — top-10 Cramér's V features from Section 2b
    SERVICES:  int    # Type of treatment service/setting (1–7)
    NOPRIOR:   int    # Prior SU treatment episodes (0, 1, 2, 3–5, 6–10, 11+)
    PSYPROB:   int    # Co-occurring mental disorder (1=Yes, 2=No)
    LIVARAG:   int    # Living arrangements (1=Homeless, 2=Dependent, 3=Independent)
    EMPLOY:    int    # Employment status (1=FT, 2=PT, 3=Unemployed, 4=Not in LF)
    ARRESTS:   int    # Arrests past 30 days (0=None, 1=Once, 2=Two or more)
    NEEDLEUSE: int    # Injection drug use (1=Yes, 2=No)
    AGECAT:    int    # Age category (1=12–14 … 7=65+)
    PSOURCE:   int    # Referral source (1–7)
    NUMSUBS:   int    # Number of substances reported (1–3)
    # Optional supporting fields
    GENDER:    Optional[int] = None   # 1=Male, 2=Female
    RACE:      Optional[int] = None   # 1–6
    EDUC:      Optional[int] = None   # 1=<9th grade … 5=College grad
    ETHNIC:    Optional[int] = None   # Ethnicity code
    MARSTAT:   Optional[int] = None   # Marital status
    VETERAN:   Optional[int] = None   # 1=Yes, 2=No
    METHUSE:   Optional[int] = None   # Max freq opioid use (1–6)
    STFIPS:    Optional[int] = None   # State FIPS code
    CBSA:      Optional[int] = None   # Census region
    HERFLG:    Optional[int] = None   # Ever used heroin (1=Yes, 2=No)
    ALCFLG:    Optional[int] = None   # Alcohol reported (1=Yes, 2=No)
    MARFLG:    Optional[int] = None   # Marijuana reported (1=Yes, 2=No)
    INHFLG:    Optional[int] = None   # Inhalants reported (1=Yes, 2=No)
    STIMFLG:   Optional[int] = None   # Stimulants (1=Yes, 2=No)
    BENZFLG:   Optional[int] = None   # Benzos (1=Yes, 2=No)
    SEDHPFLG:  Optional[int] = None   # Other sedatives (1=Yes, 2=No)
    HALLFLG:   Optional[int] = None   # Hallucinogens (1=Yes, 2=No)

class PredictionResponse(BaseModel):
    dropout_probability: float
    risk_level: str
    risk_color: str
    top_factors: list
    recommendation: str
    demo_mode: bool

# ── Build a 1-row DataFrame with correct notebook column names ─────────────────
def build_input_df(p: PatientInput) -> pd.DataFrame:
    """
    Constructs a single-row DataFrame whose column names match EXACTLY what
    the notebook's ColumnTransformer was fitted on.

    Strategy:
    - Start with all 27 training columns set to np.nan
    - Fill in the values provided by the user (mapped via CODE_TO_COL)
    - The pipeline's SimpleImputer handles any remaining NaN columns

    IMPORTANT: The ColumnTransformer splits columns into numeric vs object
    based on dtype. We preserve the correct dtypes here so the pipeline
    routes each column to the right sub-pipeline.
    """
    # Initialise all training columns as NaN
    row = {col: np.nan for col in ALL_TRAINING_COLS}

    # Map each input field to its notebook column
    field_map = {
        "SERVICES":  p.SERVICES,  "NOPRIOR": p.NOPRIOR,   "PSYPROB": p.PSYPROB,
        "LIVARAG":   p.LIVARAG,   "EMPLOY":  p.EMPLOY,    "ARRESTS": p.ARRESTS,
        "NEEDLEUSE": p.NEEDLEUSE, "AGECAT":  p.AGECAT,    "PSOURCE": p.PSOURCE,
        "NUMSUBS":   p.NUMSUBS,   "GENDER":  p.GENDER,    "RACE":    p.RACE,
        "EDUC":      p.EDUC,      "ETHNIC":  p.ETHNIC,    "MARSTAT": p.MARSTAT,
        "VETERAN":   p.VETERAN,   "METHUSE": p.METHUSE,   "STFIPS":  p.STFIPS,
        "CBSA":      p.CBSA,      "HERFLG":  p.HERFLG,    "ALCFLG":  p.ALCFLG,
        "MARFLG":    p.MARFLG,    "INHFLG":  p.INHFLG,    "STIMFLG": p.STIMFLG,
        "BENZFLG":   p.BENZFLG,   "SEDHPFLG":p.SEDHPFLG, "HALLFLG": p.HALLFLG,
    }

    for code, val in field_map.items():
        col = CODE_TO_COL.get(code)
        if col and col in row and val is not None:
            row[col] = float(val)

    df = pd.DataFrame([row])

    # Columns the notebook treated as "object" dtype (categorical)
    # These are routed to cat_pipe (mode imputation + OneHotEncoder)
    object_cols = [
        "arrests_in_past_30_days_prior_to_admission",
        "education",
        "employment_status_at_admission",
        "ethnicity",
        "gender",
        "living_arrangements_at_admission",
        "marital_status",
        "previous_substance_use_treatment_episodes",
        "referral_source",
        "co_occurring_mental_and_substance_use_disorders",
        "race",
        "veteran_status",
        "maximum_frequency_of_use_of_the_three_drugs_heroin_non_prescription_methadone_other_opiates_and_synthetics",
    ]
    for col in object_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)

    return df

# ── Demo heuristic ─────────────────────────────────────────────────────────────
def demo_predict(p: PatientInput) -> float:
    risk = 0.32
    # NOPRIOR: higher prior episodes → higher risk
    noprior_map = {0: 0.0, 1: 0.05, 2: 0.10, 3: 0.14, 4: 0.17, 5: 0.20}
    risk += noprior_map.get(p.NOPRIOR, 0.10)
    if p.PSYPROB == 1:   risk += 0.10   # co-occurring disorder
    if p.LIVARAG == 1:   risk += 0.12   # homeless
    if p.EMPLOY == 3:    risk += 0.08   # unemployed
    if p.ARRESTS >= 1:   risk += 0.07
    if p.NEEDLEUSE == 1: risk += 0.06   # injection use
    if p.SERVICES == 6:  risk += 0.05   # outpatient only (lower intensity)
    if p.SERVICES == 7:  risk -= 0.05   # OTP (structured MAT program)
    risk += np.random.uniform(-0.03, 0.03)
    return float(min(max(risk, 0.05), 0.95))

# ── Clinical factor explanations ───────────────────────────────────────────────
def explain_factors(p: PatientInput) -> list:
    factors = []
    # Risk-increasing factors
    if p.NOPRIOR >= 3:
        factors.append({"factor": "Multiple prior treatment episodes", "impact": "high", "direction": "increase"})
    if p.PSYPROB == 1:
        factors.append({"factor": "Co-occurring mental health disorder", "impact": "high", "direction": "increase"})
    if p.LIVARAG == 1:
        factors.append({"factor": "Homeless / unstable housing", "impact": "high", "direction": "increase"})
    if p.EMPLOY == 3:
        factors.append({"factor": "Currently unemployed", "impact": "medium", "direction": "increase"})
    if p.ARRESTS >= 1:
        factors.append({"factor": "Recent arrest history", "impact": "medium", "direction": "increase"})
    if p.NEEDLEUSE == 1:
        factors.append({"factor": "Injection drug use reported", "impact": "medium", "direction": "increase"})
    if p.NUMSUBS >= 2:
        factors.append({"factor": "Multiple substances reported at admission", "impact": "medium", "direction": "increase"})
    # Risk-reducing factors
    if p.SERVICES == 7:
        factors.append({"factor": "Opioid treatment program (OTP) setting", "impact": "medium", "direction": "decrease"})
    if p.EMPLOY == 1:
        factors.append({"factor": "Full-time employment", "impact": "low", "direction": "decrease"})
    if p.LIVARAG == 3:
        factors.append({"factor": "Independent / stable living", "impact": "low", "direction": "decrease"})
    return factors[:5]

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": PIPELINE is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    demo = PIPELINE is None
    try:
        if not demo:
            X_input = build_input_df(patient)
            prob = float(PIPELINE.predict_proba(X_input)[0][1])
        else:
            prob = demo_predict(patient)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if prob < 0.33:
        risk_level, risk_color = "Low Risk", "#22c55e"
        rec = ("Continue current treatment plan. Schedule regular check-ins "
               "and monitor for emerging stressors.")
    elif prob < 0.60:
        risk_level, risk_color = "Moderate Risk", "#f59e0b"
        rec = ("Consider enhanced outreach. Evaluate barriers to retention, "
               "increase session frequency, and assess MAT options.")
    else:
        risk_level, risk_color = "High Risk", "#ef4444"
        rec = ("Immediate case management intervention recommended. "
               "Prioritise MAT enrolment, stable housing support, peer recovery "
               "coaching, and address co-occurring mental health needs.")

    return PredictionResponse(
        dropout_probability=round(prob, 4),
        risk_level=risk_level,
        risk_color=risk_color,
        top_factors=explain_factors(patient),
        recommendation=rec,
        demo_mode=demo,
    )

@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost sklearn Pipeline" if PIPELINE else "Demo (heuristic)",
        "dataset": "SAMHSA TEDS-D 2018",
        "n_samples": 95432,
        "n_features": 27,
        "target": "DROPOUT (1=early dropout, 0=completed/transferred)",
        "class_balance": "~55% dropout / ~45% completed",
        "performance": {
            "ROC_AUC": 0.7160,
            "F1":      0.7263,
            "Recall":  0.8159,
            "Precision": 0.6544,
            "Accuracy":  0.6647,
        },
        "preprocessing": (
            "ColumnTransformer: "
            "numeric → median imputation + StandardScaler; "
            "object → mode imputation + OneHotEncoder(handle_unknown='ignore')"
        ),
        "training_columns": ALL_TRAINING_COLS,
        "export_from_notebook": (
            "import pickle\n"
            "with open('xgboost_pipeline.pkl', 'wb') as f:\n"
            "    pickle.dump(best_pipes['XGBoost'], f)"
        ),
    }

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
