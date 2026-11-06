import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import PlayerStats, PredictionResponse

app = FastAPI(title="FIFA 22 MLOps Prediction API", version="1.0.0")

MODEL_PATH = "models/final_fifa22_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("[FASTAPI] Model loaded.")
    except Exception as e:
        print(f"[FASTAPI ERROR] {e}")

if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print("[FASTAPI] Scaler loaded.")
    except Exception as e:
        print(f"[FASTAPI ERROR] {e}")

@app.get("/")
def root():
    return {"message": "FIFA 22 API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(player: PlayerStats):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    player_dict = player.model_dump()
    input_df = pd.DataFrame([player_dict])

    # Check if model or scaler has specific expected feature names
    expected_cols = None
    if hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
    elif hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)

    if expected_cols:
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 70.0  # sensible default for missing stat
        input_data = input_df[expected_cols]
    else:
        input_data = input_df

    if scaler is not None:
        try:
            features = scaler.transform(input_data)
        except Exception:
            features = input_data.to_numpy()
    else:
        features = input_data

    prediction = model.predict(features)[0]
    return PredictionResponse(predicted_overall=round(float(prediction), 2), status="success")