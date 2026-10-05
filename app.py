from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_fraud_model.pkl"

# Load the already-trained model. No retraining happens in the API.
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
MODEL_NAME = str(model_data.get("model_name", "XGBoost"))
THRESHOLD = float(model_data.get("threshold", 0.20))
FEATURES = list(model_data["features"])

# Compatibility repair for SimpleImputer objects saved by older
# scikit-learn versions and loaded by newer versions.
def repair_imputers(estimator):
    if isinstance(estimator, SimpleImputer):
        if not hasattr(estimator, "_fill_dtype") and hasattr(estimator, "statistics_"):
            estimator._fill_dtype = estimator.statistics_.dtype
        if not hasattr(estimator, "_fit_dtype") and hasattr(estimator, "statistics_"):
            estimator._fit_dtype = estimator.statistics_.dtype

    if hasattr(estimator, "steps"):
        for _, step in estimator.steps:
            repair_imputers(step)

    if hasattr(estimator, "transformers_"):
        for _, transformer, _ in estimator.transformers_:
            repair_imputers(transformer)

    if hasattr(estimator, "transformer_list"):
        for _, transformer in estimator.transformer_list:
            repair_imputers(transformer)

repair_imputers(model)

app = FastAPI(
    title="Vehicle Insurance Claim Fraud Detection API",
    description="REST API for the trained XGBoost vehicle insurance fraud detection model.",
    version="1.0.0"
)

# Useful when a frontend is connected later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_METRICS = {
    "accuracy": 95.49,
    "precision": 60.55,
    "recall": 71.35,
    "f1_score": 65.51,
    "roc_auc": 96.30,
    "threshold": THRESHOLD,
    "confusion_matrix": {
        "true_negative": 2813,
        "false_positive": 86,
        "false_negative": 53,
        "true_positive": 132
    }
}

NUMERIC_FIELDS = {
    "WeekOfMonth",
    "WeekOfMonthClaimed",
    "Age",
    "PolicyNumber",
    "RepNumber",
    "Deductible",
    "DriverRating",
    "Year"
}


@app.get("/")
def root():
    return {
        "service": "Vehicle Insurance Claim Fraud Detection API",
        "status": "running",
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "docs": "/docs"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": MODEL_NAME,
        "threshold": THRESHOLD
    }


@app.get("/api/model-info")
def model_info():
    return {
        "model": MODEL_NAME,
        "metrics": MODEL_METRICS,
        "feature_count": len(FEATURES),
        "features": FEATURES
    }


@app.get("/api/features")
def features():
    return {
        "count": len(FEATURES),
        "features": FEATURES,
        "numeric_features": sorted(NUMERIC_FIELDS),
        "categorical_features": [
            feature for feature in FEATURES if feature not in NUMERIC_FIELDS
        ]
    }


def prepare_input(payload: Dict[str, Any]) -> pd.DataFrame:
    missing = [feature for feature in FEATURES if feature not in payload]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required fields",
                "missing_fields": missing
            }
        )

    row = {feature: payload[feature] for feature in FEATURES}

    for field in NUMERIC_FIELDS:
        value = row[field]

        if value is None or value == "":
            row[field] = np.nan
            continue

        try:
            number = float(value)

            if not np.isfinite(number):
                raise ValueError

            row[field] = number

        except (TypeError, ValueError):
            raise HTTPException(
                status_code=422,
                detail=f"Field '{field}' must contain a valid number."
            )

    return pd.DataFrame([row], columns=FEATURES)


@app.post("/api/predict")
def predict(payload: Dict[str, Any]):
    """
    Predict whether an insurance claim is potentially fraudulent.

    The saved model's fraud probability is compared against the tuned
    classification threshold of 0.20.
    """
    try:
        data = prepare_input(payload)

        # Class 1 = fraudulent claim.
        fraud_probability = float(model.predict_proba(data)[0, 1])

        prediction = int(fraud_probability >= THRESHOLD)

        if prediction == 1:
            result = "Potentially Fraudulent"
            risk_level = "High"
        else:
            result = "Likely Genuine"
            risk_level = "Low"

        return {
            "success": True,
            "prediction": prediction,
            "result": result,
            "risk_level": risk_level,
            "fraud_probability": round(fraud_probability, 6),
            "fraud_probability_percent": round(fraud_probability * 100, 2),
            "threshold": THRESHOLD,
            "threshold_percent": round(THRESHOLD * 100, 2),
            "model": MODEL_NAME
        }

    except HTTPException:
        raise

    except Exception:
        import logging
        logging.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend console."
        )
