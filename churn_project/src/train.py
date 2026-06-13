"""
train.py
---------
End-to-end training pipeline for the customer churn prediction model.

Steps:
1. Load dataset
2. Feature engineering (derive behavioral ratio features)
3. Train/test split (stratified)
4. Establish a BASELINE model (raw features, plain Logistic Regression)
5. Scale features and compare candidate models (LogReg, RandomForest,
   GradientBoosting) with engineered features + class balancing, using
   5-fold cross-validation
6. Hyperparameter-tune the best model with GridSearchCV
7. Evaluate the tuned model on the held-out test set
8. Save the trained model, scaler, and evaluation plots
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

DATA_PATH = "/home/claude/churn_project/data/customer_data.csv"
MODEL_DIR = "/home/claude/churn_project/models"
PLOT_DIR = "/home/claude/churn_project/plots"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

raw_features = [
    "tenure_months", "monthly_charges", "total_purchases", "avg_order_value",
    "num_support_tickets", "days_since_last_purchase", "is_premium_member",
    "num_returns", "app_sessions_per_week", "email_open_rate", "discount_usage_rate",
]

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
df["purchase_frequency"] = df["total_purchases"] / (df["tenure_months"] + 1)
df["support_ticket_rate"] = df["num_support_tickets"] / (df["tenure_months"] + 1)
df["return_rate"] = df["num_returns"] / (df["total_purchases"] + 1)
df["recency_ratio"] = df["days_since_last_purchase"] / (df["tenure_months"] * 30 + 1)
df["engagement_score"] = df["app_sessions_per_week"] * df["email_open_rate"]

engineered_features = [
    "purchase_frequency", "support_ticket_rate", "return_rate",
    "recency_ratio", "engagement_score",
]
all_features = raw_features + engineered_features

X_all = df[all_features]
y = df["churn"]

# ---------------------------------------------------------------------------
# 3. Train/test split (shared split for fair baseline vs. improved comparison)
# ---------------------------------------------------------------------------
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 4. BASELINE: plain Logistic Regression on raw (unscaled, unbalanced) features
# ---------------------------------------------------------------------------
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train_all[raw_features], y_train)
baseline_pred = baseline_model.predict(X_test_all[raw_features])
baseline_proba = baseline_model.predict_proba(X_test_all[raw_features])[:, 1]

baseline_metrics = {
    "accuracy": accuracy_score(y_test, baseline_pred),
    "precision": precision_score(y_test, baseline_pred, zero_division=0),
    "recall": recall_score(y_test, baseline_pred, zero_division=0),
    "f1_score": f1_score(y_test, baseline_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, baseline_proba),
}
print("=== Baseline (raw features, plain Logistic Regression) ===")
for k, v in baseline_metrics.items():
    print(f"{k}: {v:.4f}")

# ---------------------------------------------------------------------------
# 5. Scale features (raw + engineered) and compare candidate models via CV
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test_all)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

cv_results = {}
print("\n=== Candidate model comparison (5-fold CV ROC-AUC, engineered features) ===")
for name, model in candidates.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
    cv_results[name] = {"mean_auc": float(scores.mean()), "std_auc": float(scores.std())}
    print(f"{name}: CV ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

best_model_name = max(cv_results, key=lambda k: cv_results[k]["mean_auc"])
print(f"\nBest baseline model: {best_model_name}")

# ---------------------------------------------------------------------------
# 6. Hyperparameter tuning on the best model
# ---------------------------------------------------------------------------
if best_model_name == "Random Forest":
    base_model = RandomForestClassifier(random_state=42, class_weight="balanced")
    param_grid = {
        "n_estimators": [150, 250, 350],
        "max_depth": [6, 10, 14],
        "min_samples_leaf": [1, 3, 5],
    }
elif best_model_name == "Gradient Boosting":
    base_model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    }
else:
    base_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
    }

grid = GridSearchCV(base_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
print(f"\nBest params: {grid.best_params_}")
print(f"Best CV ROC-AUC after tuning: {grid.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 7. Evaluate tuned model on held-out test set
# ---------------------------------------------------------------------------
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}

print("\n=== Final Tuned Model - Test Set Performance ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n=== Improvement over baseline ===")
for k in metrics:
    delta = metrics[k] - baseline_metrics[k]
    print(f"{k}: {baseline_metrics[k]:.4f} -> {metrics[k]:.4f}  (delta {delta:+.4f})")

# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
# Confusion matrix
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
ax.set_title(f"Confusion Matrix - {best_model_name} (Tuned)")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ROC curve (baseline vs tuned)
fpr, tpr, _ = roc_curve(y_test, y_proba)
bfpr, btpr, _ = roc_curve(y_test, baseline_proba)
plt.figure(figsize=(5.5, 4.5))
plt.plot(fpr, tpr, label=f"Tuned {best_model_name} (AUC={metrics['roc_auc']:.3f})", linewidth=2)
plt.plot(bfpr, btpr, label=f"Baseline LogReg (AUC={baseline_metrics['roc_auc']:.3f})",
         linestyle="--", linewidth=2)
plt.plot([0, 1], [0, 1], "k:", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Baseline vs. Tuned Model")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/roc_curve.png", dpi=150)
plt.close()

# Feature importance / coefficients
plt.figure(figsize=(7, 5))
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.barh([all_features[i] for i in order][::-1], importances[order][::-1], color="#4C72B0")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance - {best_model_name}")
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])
    order = np.argsort(importances)[::-1]
    plt.barh([all_features[i] for i in order][::-1], importances[order][::-1], color="#4C72B0")
    plt.xlabel("|Coefficient| (standardized)")
    plt.title(f"Feature Importance - {best_model_name}")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Save model, scaler, metadata
# ---------------------------------------------------------------------------
joblib.dump(best_model, f"{MODEL_DIR}/churn_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump({
        "model_name": best_model_name,
        "best_params": grid.best_params_,
        "raw_features": raw_features,
        "engineered_features": engineered_features,
        "all_features": all_features,
        "baseline_metrics": baseline_metrics,
        "final_metrics": metrics,
        "cv_results": cv_results,
        "dataset_size": len(df),
        "churn_rate": float(df["churn"].mean()),
    }, f, indent=2)

print("\nModel, scaler, and metadata saved to /models")
print("Plots saved to /plots")
