"""
app.py
-------
Flask REST API for real-time churn predictions with churn.csv feature support.
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "scaler.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

app = Flask(__name__)

def load_artifacts():
    if not (MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and METADATA_PATH.exists()):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    return model, preprocessor, meta

model, preprocessor, metadata = load_artifacts()

@app.route("/", methods=["GET"])
def home():
    if metadata is None:
        return jsonify({"status": "Model artifacts missing. Run src/train.py first."}), 503
    return jsonify({
        "service": "E-Commerce Customer Churn Prediction API",
        "model": metadata["model_name"],
        "test_roc_auc": metadata["final_metrics"]["roc_auc"],
        "usage": "POST /predict with customer JSON payload"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or preprocessor is None:
        return jsonify({"error": "Model missing. Train model first."}), 500

    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON request body."}), 400

    num_cols = metadata["numeric_features"]
    cat_cols = metadata["categorical_features"]
    expected_cols = num_cols + cat_cols

    # Convert payload dictionary to DataFrame
    input_df = pd.DataFrame([payload])

    # Clean input data
    if "avg_frequency_login_days" in input_df.columns:
        input_df["avg_frequency_login_days"] = pd.to_numeric(input_df["avg_frequency_login_days"], errors="coerce")

    for c in num_cols:
        if c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce")
            input_df[c] = input_df[c].apply(lambda x: np.nan if pd.notnull(x) and x < 0 else x)

    # Ensure all required features are present
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[expected_cols]

    try:
        X_trans = preprocessor.transform(input_df)
        churn_proba = float(model.predict_proba(X_trans)[0, 1])
        churn_pred = int(model.predict(X_trans)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 400

    return jsonify({
        "churn_prediction": churn_pred,
        "churn_probability": round(churn_proba, 4),
        "risk_level": (
            "High" if churn_proba >= 0.6 else
            "Medium" if churn_proba >= 0.3 else
            "Low"
        )
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
