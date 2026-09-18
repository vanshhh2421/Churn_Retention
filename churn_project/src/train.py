"""
train.py
---------
End-to-end training pipeline tailored for churn.csv.
Cleans raw data, handles invalid values ('Error', -999), encodes
categorical variables, trains a balanced model, and saves artifacts.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "churn.csv"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Place churn.csv in data/ directory.")

# 1. Load Data
df = pd.read_csv(DATA_PATH)

# Drop non-predictive identifier columns
drop_cols = ["Unnamed: 0", "security_no", "referral_id", "joining_date", "last_visit_time"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 2. Data Cleaning
# Convert avg_frequency_login_days to numeric (coercing 'Error' strings to NaN)
if "avg_frequency_login_days" in df.columns:
    df["avg_frequency_login_days"] = pd.to_numeric(df["avg_frequency_login_days"], errors="coerce")

# Clean sentinel/invalid negative values
num_clean_cols = ["days_since_last_login", "avg_time_spent", "points_in_wallet", "avg_frequency_login_days"]
for col in num_clean_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: np.nan if pd.notnull(x) and x < 0 else x)

# Replace '?' string placeholders in categorical columns with NaN
cat_cols_raw = df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols_raw:
    df[col] = df[col].replace("?", np.nan)

# Separate features and target
TARGET = "churn_risk_score"
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 3. Build Preprocessing Transformers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit Preprocessor
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# 5. Model Training & Tuning
classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}

grid = GridSearchCV(classifier, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train_trans, y_train)

best_model = grid.best_estimator_

# 6. Evaluation
y_pred = best_model.predict(X_test_trans)
y_proba = best_model.predict_proba(X_test_trans)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba)
}

# 7. Plots
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["No Churn", "Churn"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No Churn", "Churn"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Churn Model")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5.5, 4.5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={metrics['roc_auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k:", alpha=0.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(); plt.tight_layout()
plt.savefig(PLOT_DIR / "roc_curve.png", dpi=150)
plt.close()

# 8. Save Artifacts
joblib.dump(best_model, MODEL_DIR / "churn_model.pkl")
joblib.dump(preprocessor, MODEL_DIR / "scaler.pkl")  # Saves fitted preprocessor pipeline

metadata = {
    "model_name": "Logistic Regression (Tuned)",
    "best_params": grid.best_params_,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "final_metrics": metrics,
    "dataset_size": len(df),
    "churn_rate": float(df[TARGET].mean())
}

with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Training completed successfully.")
print(f"Metrics: AUC={metrics['roc_auc']:.4f}, Recall={metrics['recall']:.4f}, Accuracy={metrics['accuracy']:.4f}")
