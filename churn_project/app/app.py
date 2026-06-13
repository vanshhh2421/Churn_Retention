"""
app.py
-------
A lightweight Flask REST API that serves real-time churn predictions
using the trained model and scaler.

Run:
    python3 app/app.py

Then send a POST request to /predict, e.g.:

curl -X POST http://127.0.0.1:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
        "tenure_months": 5,
        "monthly_charges": 70,
        "total_purchases": 3,
        "avg_order_value": 40,
        "num_support_tickets": 4,
        "days_since_last_purchase": 60,
        "is_premium_member": 0,
        "num_returns": 2,
        "app_sessions_per_week": 1,
        "email_open_rate": 0.1,
        "discount_usage_rate": 0.05
      }'
"""

import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

MODEL_PATH = "/home/claude/churn_project/models/churn_model.pkl"
SCALER_PATH = "/home/claude/churn_project/models/scaler.pkl"
METADATA_PATH = "/home/claude/churn_project/models/metadata.json"

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

RAW_FEATURES = metadata["raw_features"]
ALL_FEATURES = metadata["all_features"]


def engineer_features(payload):
    """Compute the same engineered features used during training."""
    row = {f: float(payload[f]) for f in RAW_FEATURES}

    row["purchase_frequency"] = row["total_purchases"] / (row["tenure_months"] + 1)
    row["support_ticket_rate"] = row["num_support_tickets"] / (row["tenure_months"] + 1)
    row["return_rate"] = row["num_returns"] / (row["total_purchases"] + 1)
    row["recency_ratio"] = row["days_since_last_purchase"] / (row["tenure_months"] * 30 + 1)
    row["engagement_score"] = row["app_sessions_per_week"] * row["email_open_rate"]

    return np.array([[row[f] for f in ALL_FEATURES]])


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Customer Churn Prediction API",
        "model": metadata["model_name"],
        "test_roc_auc": metadata["final_metrics"]["roc_auc"],
        "usage": "POST /predict with a JSON body containing: " + ", ".join(RAW_FEATURES),
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    missing = [f for f in RAW_FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        features = engineer_features(payload)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    features_scaled = scaler.transform(features)
    churn_proba = float(model.predict_proba(features_scaled)[0, 1])
    churn_pred = int(model.predict(features_scaled)[0])

    return jsonify({
        "churn_prediction": churn_pred,
        "churn_probability": round(churn_proba, 4),
        "risk_level": (
            "High" if churn_proba >= 0.6 else
            "Medium" if churn_proba >= 0.3 else
            "Low"
        ),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
