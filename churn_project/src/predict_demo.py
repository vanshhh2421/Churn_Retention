"""
predict_demo.py
-----------------
Standalone CLI script for rapid local testing using churn.csv schema.
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

if not (MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and METADATA_PATH.exists()):
    raise FileNotFoundError("Artifacts missing. Run `python3 src/train.py` first.")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

expected_cols = metadata["numeric_features"] + metadata["categorical_features"]

examples = {
    "High Risk Customer": {
        "age": 44,
        "gender": "F",
        "region_category": "Town",
        "membership_category": "No Membership",
        "joined_through_referral": "Yes",
        "preferred_offer_types": "Gift Vouchers/Coupons",
        "medium_of_operation": "Desktop",
        "internet_option": "Wi-Fi",
        "days_since_last_login": 14,
        "avg_time_spent": 516.16,
        "avg_transaction_value": 21027.0,
        "avg_frequency_login_days": 22.0,
        "points_in_wallet": 500.69,
        "used_special_discount": "No",
        "offer_application_preference": "Yes",
        "past_complaint": "Yes",
        "complaint_status": "Solved in Follow-up",
        "feedback": "Poor Website"
    },
    "Low Risk Customer": {
        "age": 18,
        "gender": "F",
        "region_category": "Village",
        "membership_category": "Platinum Membership",
        "joined_through_referral": "No",
        "preferred_offer_types": "Gift Vouchers/Coupons",
        "medium_of_operation": "Desktop",
        "internet_option": "Wi-Fi",
        "days_since_last_login": 17,
        "avg_time_spent": 300.63,
        "avg_transaction_value": 53005.25,
        "avg_frequency_login_days": 17.0,
        "points_in_wallet": 781.75,
        "used_special_discount": "Yes",
        "offer_application_preference": "Yes",
        "past_complaint": "No",
        "complaint_status": "Not Applicable",
        "feedback": "Products always in Stock"
    }
}

for label, payload in examples.items():
    df = pd.DataFrame([payload])
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected_cols]

    X_trans = preprocessor.transform(df)
    proba = float(model.predict_proba(X_trans)[0, 1])
    pred = int(model.predict(X_trans)[0])
    risk = "High" if proba >= 0.6 else "Medium" if proba >= 0.3 else "Low"

    print(f"\n{label}:")
    print(f"  -> Prediction: {pred} | Probability: {proba:.4f} | Risk Level: {risk}")
