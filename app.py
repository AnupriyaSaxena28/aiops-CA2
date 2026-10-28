from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
FEATURE_NAMES_PATH = BASE_DIR / "feature_names.json"

if not MODEL_PATH.exists():
    raise RuntimeError(
        "model.joblib was not found. Run the Colab notebook and copy model.joblib "
        "into the project folder."
    )

model = joblib.load(MODEL_PATH)
with FEATURE_NAMES_PATH.open("r", encoding="utf-8") as f:
    FEATURE_NAMES = __import__("json").load(f)

app = FastAPI(
    title="Spambase Spam Classification API",
    description="Predicts whether a numerical Spambase feature vector is spam or legitimate email.",
    version="1.0.0",
)


class EmailFeatures(BaseModel):
    features: list[float] = Field(
        ...,
        description="Exactly 57 numeric features in the same order as the Spambase CSV columns (excluding class).",
        examples=[[0.0] * 57],
    )

    @field_validator("features")
    @classmethod
    def check_feature_count(cls, values):
        if len(values) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, received {len(values)}.")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("All feature values must be finite numbers.")
        return values


@app.get("/")
def root():
    return {
        "message": "Spambase Spam Classification API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "feature_count": len(FEATURE_NAMES)}


@app.post("/predict")
def predict_email(payload: EmailFeatures):
    values = pd.DataFrame([payload.features], columns=FEATURE_NAMES, dtype=float)
    prediction = int(model.predict(values)[0])
    probability = model.predict_proba(values)[0]
    spam_probability = float(probability[1])
    return {
        "prediction": prediction,
        "label": "spam" if prediction == 1 else "ham",
        "spam_probability": round(spam_probability, 4),
        "message": "Likely spam" if prediction == 1 else "Likely legitimate email",
    }
