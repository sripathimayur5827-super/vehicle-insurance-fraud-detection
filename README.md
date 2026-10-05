# Vehicle Insurance Claim Fraud Detection — FastAPI Backend

Backend only. This project uses the existing `best_fraud_model.pkl`.
It does not retrain the model.

## 1. Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run

```bash
uvicorn app:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

## 3. API endpoints

### Health
`GET /api/health`

### Model information
`GET /api/model-info`

### Model feature list
`GET /api/features`

### Prediction
`POST /api/predict`

The prediction request must contain the exact 32 feature names stored inside the pickle.

## Model logic

The saved pipeline produces the fraud probability:

```python
fraud_probability = model.predict_proba(data)[0, 1]
```

The final tuned threshold is `0.20`:

```python
prediction = int(fraud_probability >= 0.20)
```

Therefore:

- `prediction = 0` → Likely Genuine
- `prediction = 1` → Potentially Fraudulent

## Example prediction response

```json
{
  "success": true,
  "prediction": 1,
  "result": "Potentially Fraudulent",
  "risk_level": "High",
  "fraud_probability": 0.7342,
  "fraud_probability_percent": 73.42,
  "threshold": 0.2,
  "threshold_percent": 20.0,
  "model": "XGBoost"
}
```

## Important

The pickle contains the trained preprocessing/model pipeline, so the API sends the input through the saved pipeline rather than manually recreating preprocessing.

A small `SimpleImputer` compatibility repair is included because the saved pickle was produced with an older scikit-learn fitted state.
