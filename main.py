import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aapka Hugging Face Model Repo ID
HF_REPO_ID = "celestial999/disease-model"

print("⏳ Downloading 5 trained models from Hugging Face Repo...")
m_heart_path = hf_hub_download(repo_id=HF_REPO_ID, filename="heart_model.pkl")
m_diab_path = hf_hub_download(repo_id=HF_REPO_ID, filename="diabetes_model.pkl")
m_stroke_path = hf_hub_download(repo_id=HF_REPO_ID, filename="stroke_model.pkl")
m_htn_path = hf_hub_download(repo_id=HF_REPO_ID, filename="hypertension_model.pkl")
m_meta_path = hf_hub_download(repo_id=HF_REPO_ID, filename="metabolic_model.pkl")

# Load 5 Machine Learning Models
heart_model = joblib.load(m_heart_path)
diabetes_model = joblib.load(m_diab_path)
stroke_model = joblib.load(m_stroke_path)
hypertension_model = joblib.load(m_htn_path)
metabolic_model = joblib.load(m_meta_path)
print("✅ All 5 Production Models Loaded Successfully!")

@app.get("/")
def health():
    return {"status": "5-Tier Multi-Disease ML Engine is Live on Render!"}

@app.post("/predict")
async def predict_full_suite(request: Request):
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        body = {}

    age = int(body.get('age') or 45)
    bmi = float(body.get('bmi') or 24.5)
    bp = str(body.get('blood_pressure_level') or body.get('bp') or 'Normal').lower()
    smoking = str(body.get('smoking_habit') or 'Non-Smoker').lower()
    alcohol = str(body.get('alcohol_consumption') or 'None').lower()
    activity = str(body.get('physical_activity') or 'Moderate').lower()
    family_hist = [str(x).lower() for x in (body.get('family_history') or [])] if isinstance(body.get('family_history'), list) else [str(body.get('family_history') or '').lower()]
    sugar = str(body.get('blood_sugar') or 'Normal').lower()

    # Exact Feature Parsing
    smoking_num = 1 if any(x in smoking for x in ["regular", "occasional", "yes"]) else 0
    bp_high_num = 1 if any(x in bp for x in ["stage", "elevated", "140", "yes"]) else 0
    sugar_high_num = 1 if any(x in sugar for x in ["high", "elevated", "mg"]) and not ("9" in sugar or "8" in sugar or "normal" in sugar) else 0
    family_heart_num = 1 if any("heart" in x for x in family_hist) else 0
    family_diab_num = 1 if any("diabet" in x for x in family_hist) else 0
    sedentary_num = 1 if any(x in activity for x in ["sedentary", "low", "no"]) else 0
    alcohol_num = 15 if "regular" in alcohol else 3 if "occasional" in alcohol else 0

    # DataFrames for sklearn consistency
    df_heart = pd.DataFrame([{'age': age, 'bmi': bmi, 'smoking': smoking_num, 'bp_high': bp_high_num, 'family_heart': family_heart_num}])
    df_diab = pd.DataFrame([{'age': age, 'bmi': bmi, 'sugar_high': sugar_high_num, 'family_diab': family_diab_num, 'sedentary': sedentary_num}])
    df_stroke = pd.DataFrame([{'age': age, 'bp_high': bp_high_num, 'smoking': smoking_num, 'bmi': bmi}])
    df_htn = pd.DataFrame([{'age': age, 'bmi': bmi, 'smoking': smoking_num, 'alcohol': alcohol_num, 'sedentary': sedentary_num}])
    df_meta = pd.DataFrame([{'bmi': bmi, 'alcohol': alcohol_num, 'sugar_high': sugar_high_num, 'bp_high': bp_high_num}])

    # Pure ML Model Predictions
    p_heart = float(heart_model.predict_proba(df_heart)[0][1])
    p_diab = float(diabetes_model.predict_proba(df_diab)[0][1])
    p_stroke = float(stroke_model.predict_proba(df_stroke)[0][1])
    p_htn = float(hypertension_model.predict_proba(df_htn)[0][1])
    p_meta = float(metabolic_model.predict_proba(df_meta)[0][1])

    detected = []
    recs = []

    if p_heart >= 0.38:
        detected.append(f"🫀 Coronary Heart Disease (Confidence: {round(p_heart*100, 1)}%)")
        recs.append("Preventive Cardiology Consult (ECG / Lipid Profile) recommended.")

    if p_diab >= 0.38:
        detected.append(f"🩸 Type-2 Diabetes (Confidence: {round(p_diab*100, 1)}%)")
        recs.append("Fasting Blood Sugar and HbA1c screening recommended.")

    if p_stroke >= 0.38:
        detected.append(f"🧠 Stroke & Vascular Blockage (Confidence: {round(p_stroke*100, 1)}%)")
        recs.append("Arterial blood pressure control and vascular screening advised.")

    if p_htn >= 0.40:
        detected.append(f"🩺 Chronic Hypertension (Confidence: {round(p_htn*100, 1)}%)")
        recs.append("Adopt a low-sodium DASH diet and track daily blood pressure.")

    if p_meta >= 0.40:
        detected.append(f"⚖️ Metabolic Syndrome (Confidence: {round(p_meta*100, 1)}%)")
        recs.append("Maintain a caloric deficit and minimum 150 mins weekly aerobic exercise.")

    overall_max_prob = max(p_heart, p_diab, p_stroke, p_htn, p_meta) * 100.0

    if not detected:
        risk_level = "Low Risk"
        disease_summary = "Healthy Baseline (All 5 ML clinical models evaluated negative)"
        recs = [
            "All vital parameters align with healthy clinical baseline ranges.",
            "Maintain current balanced nutrition and routine annual checkups."
        ]
    elif overall_max_prob < 55.0:
        risk_level = "Moderate Risk"
        disease_summary = "\n• ".join(detected)
    else:
        risk_level = "High Risk"
        disease_summary = "\n• ".join(detected)

    return {
        "risk_percentage": f"{round(overall_max_prob, 1)}%",
        "risk_level": risk_level,
        "target_disease": disease_summary,
        "recommendations": recs[:4]
    }
