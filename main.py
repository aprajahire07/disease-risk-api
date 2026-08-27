import os
import urllib.request
import joblib
import pandas as pd
import numpy as np
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aapka Hugging Face Direct Download Link
MODEL_URL = "https://huggingface.co/celestial999/disease-model/resolve/main/lifestyle_form_model.pkl"
MODEL_PATH = "lifestyle_form_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model Downloaded successfully!")

model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"status": "Disease API is Live 24/7!"}

@app.post("/predict")
async def predict_risk(request: Request):
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        body = {}

    age = int(body.get('age') or 45)
    gender = str(body.get('gender') or 'Male')
    bmi = float(body.get('bmi') or 24.5)
    bp = str(body.get('blood_pressure_level') or body.get('bp') or 'Normal')
    smoking = str(body.get('smoking_habit') or 'Non-Smoker')
    alcohol = str(body.get('alcohol_consumption') or 'None')
    activity = str(body.get('physical_activity') or 'Moderate')
    family_hist = body.get('family_history') or []
    if isinstance(family_hist, str): 
        family_hist = [family_hist]
    sugar = str(body.get('blood_sugar') or 'Normal')

    age_cat = "18-24" if age < 25 else "25-39" if age < 40 else "40-54" if age < 55 else "55-64" if age < 65 else "65+"
    smoking_flag = "Yes" if any(x in smoking.lower() for x in ["regular", "occasional", "yes"]) else "No"
    exercise_flag = "Yes" if any(x in activity.lower() for x in ["active", "moderate", "yes"]) else "No"
    alcohol_val = 15 if "regular" in alcohol.lower() else 3 if "occasional" in alcohol.lower() else 0
    diabetes_flag = "Yes" if any("diabet" in str(x).lower() for x in family_hist) or ("normal" not in sugar.lower() and sugar != "") else "No"

    input_df = pd.DataFrame([{
        'Sex': "Male" if "m" in gender.lower() and "fe" not in gender.lower() else "Female",
        'Age_Category': age_cat,
        'BMI': bmi,
        'Smoking_History': smoking_flag,
        'Alcohol_Consumption': alcohol_val,
        'Exercise': exercise_flag,
        'Diabetes': diabetes_flag,
        'Depression': 'No'
    }])

    raw_model_prob = float(model.predict_proba(input_df)[0][1])
    
    detected_diseases = []
    recommendations_list = []
    accumulated_risk_points = 0

    has_heart_hist = any("heart" in str(x).lower() for x in family_hist)
    is_high_bp = "stage 2" in bp.lower() or "140" in bp
    if has_heart_hist or is_high_bp or (age >= 55 and smoking_flag == "Yes"):
        detected_diseases.append("🫀 Coronary Heart Disease (Dil ki naso me strain)")
        recommendations_list.append("Schedule an ECG, Lipid Profile test & reduce dietary sodium.")
        accumulated_risk_points += 30

    if "stage 2" in bp.lower() or "140" in bp:
        detected_diseases.append("🩺 Stage-2 Hypertension (Severe High BP)")
        accumulated_risk_points += 25
    elif "stage 1" in bp.lower() or "elevated" in bp.lower():
        detected_diseases.append("🩺 Pre-Hypertension (Borderline High BP)")
        recommendations_list.append("Daily blood pressure monitoring required.")
        accumulated_risk_points += 15

    has_diab_hist = any("diabet" in str(x).lower() for x in family_hist)
    is_sugar_high = any(x in sugar.lower() for x in ["high", "elevated", "mg"]) and not ("9" in sugar or "8" in sugar or "normal" in sugar.lower())
    if is_sugar_high or has_diab_hist or (bmi >= 30 and age >= 45):
        detected_diseases.append("🩸 Type-2 Diabetes (High blood sugar & insulin resistance)")
        recommendations_list.append("Get a Fasting Blood Glucose and HbA1c test done.")
        accumulated_risk_points += 30

    if "regular" in smoking.lower() and (is_high_bp or age >= 50):
        detected_diseases.append("🧠 Stroke & Vascular Blockage (Dimag ki naso me rukawat)")
        recommendations_list.append("Immediate smoking cessation and vascular health screening.")
        accumulated_risk_points += 20

    if bmi >= 30:
        detected_diseases.append("⚖️ Obesity & Metabolic Syndrome (Extra fat & metabolic stress)")
        recommendations_list.append("Aim for minimum 150 mins cardio per week with a low-carb diet.")
        accumulated_risk_points += 20
    elif bmi >= 25:
        accumulated_risk_points += 10

    if "sedentary" in activity.lower():
        accumulated_risk_points += 10
    if "regular" in alcohol.lower():
        accumulated_risk_points += 15
        recommendations_list.append("Reduce alcohol intake to prevent liver & cardiac strain.")

    final_score = min(96.0, max(4.0, (accumulated_risk_points * 0.7) + (raw_model_prob * 100 * 0.3)))

    if final_score < 30:
        level = "Low Risk"
        disease_summary = "Healthy Baseline (No high-risk disease detected)"
        recommendations_list = [
            "Your overall lifestyle parameters are in good range.",
            "Maintain current balanced nutrition and routine annual checkups."
        ]
    elif final_score < 65:
        level = "Moderate Risk"
        disease_summary = "\n• ".join(detected_diseases) if detected_diseases else "Early Lifestyle Strain"
    else:
        level = "High Risk"
        disease_summary = "\n• ".join(detected_diseases) if detected_diseases else "Multiple Risk Factors Present"

    return {
        "risk_percentage": f"{round(final_score, 1)}%",
        "risk_level": level,
        "target_disease": disease_summary,
        "recommendations": recommendations_list[:4] if recommendations_list else ["Maintain active routine."]
    }