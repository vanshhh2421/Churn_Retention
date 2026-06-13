"""
predict_demo.py
-----------------
Standalone demo script that loads the trained model and scaler and
runs predictions on a few example customers. Useful for quickly
verifying the model works without spinning up the Flask server.
"""

import json
import joblib
import numpy as np

MODEL_PATH = "/home/claude/churn_project/models/churn_model.pkl"
SCALER_PATH = "/home/claude/churn_project/models/scaler.pkl"
METADATA_PATH = "/home/claude/churn_project/models/metadata.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

RAW_FEATURES = metadata["raw_features"]
ALL_FEATURES = metadata["all_features"]


def engineer_features(payload):
    row = {f: float(payload[f]) for f in RAW_FEATURES}
    row["purchase_frequency"] = row["total_purchases"] / (row["tenure_months"] + 1)
    row["support_ticket_rate"] = row["num_support_tickets"] / (row["tenure_months"] + 1)
    row["return_rate"] = row["num_returns"] / (row["total_purchases"] + 1)
    row["recency_ratio"] = row["days_since_last_purchase"] / (row["tenure_months"] * 30 + 1)
    row["engagement_score"] = row["app_sessions_per_week"] * row["email_open_rate"]
    return np.array([[row[f] for f in ALL_FEATURES]])


examples = {
    "Likely to churn (low engagement, many tickets, inactive)": {
        "tenure_months": 5, "monthly_charges": 70, "total_purchases": 3,
        "avg_order_value": 40, "num_support_tickets": 4,
        "days_since_last_purchase": 60, "is_premium_member": 0,
        "num_returns": 2, "app_sessions_per_week": 1,
        "email_open_rate": 0.1, "discount_usage_rate": 0.05,
    },
    "Loyal customer (long tenure, premium, engaged)": {
        "tenure_months": 48, "monthly_charges": 35, "total_purchases": 60,
        "avg_order_value": 35, "num_support_tickets": 0,
        "days_since_last_purchase": 2, "is_premium_member": 1,
        "num_returns": 0, "app_sessions_per_week": 10,
        "email_open_rate": 0.7, "discount_usage_rate": 0.1,
    },
}

for label, payload in examples.items():
    X = engineer_features(payload)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]
    risk = "High" if proba >= 0.6 else "Medium" if proba >= 0.3 else "Low"
    print(f"\n{label}")
    print(f"  -> churn_prediction: {pred}, churn_probability: {proba:.3f}, risk_level: {risk}")
