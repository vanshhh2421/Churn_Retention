# Customer Churn Prediction — E-Commerce

An end-to-end machine learning pipeline that predicts which e-commerce customers
are at risk of churning, so retention teams can intervene early. Built with
**scikit-learn**, deployed via a **Flask REST API**.

---

## 📌 Problem Statement

Customer retention is far cheaper than acquisition. This project builds a
binary classifier that flags customers likely to churn based on their
behavioral and transactional history (tenure, spend, support interactions,
engagement, etc.), enabling proactive retention campaigns.

---

## 🗂️ Project Structure

```
churn_project/
├── data/
│   ├── generate_data.py      # Synthetic e-commerce dataset generator
│   └── customer_data.csv     # 10,000-row dataset (11 raw features + label)
├── src/
│   ├── train.py               # Full training pipeline (see below)
│   └── predict_demo.py        # Quick inference demo
├── models/
│   ├── churn_model.pkl        # Trained, tuned model (ready to use)
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── metadata.json          # Feature lists, metrics, best hyperparameters
├── plots/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── app/
│   └── app.py                  # Flask REST API for real-time predictions
└── requirements.txt
```

---

## ⚙️ Pipeline Overview

1. **Data**: 10,000 synthetic customer records with 11 behavioral/transactional
   features (tenure, monthly spend, purchase history, support tickets,
   engagement metrics, etc.) and a churn label (~19.4% positive class —
   realistic for retail churn).
2. **Feature Engineering**: 5 additional ratio-based features were engineered
   (purchase frequency, support ticket rate, return rate, recency ratio,
   engagement score) to capture customer *behavior trends* rather than raw
   counts.
3. **Baseline**: Plain Logistic Regression on raw features only, no scaling,
   no class balancing.
4. **Model Selection**: Logistic Regression, Random Forest, and Gradient
   Boosting were compared using **5-fold cross-validation** (ROC-AUC) on the
   scaled, engineered feature set with class-weight balancing to address
   class imbalance.
5. **Hyperparameter Tuning**: The best-performing model family was tuned via
   `GridSearchCV` (5-fold CV).
6. **Evaluation**: Final model evaluated on a held-out 20% test set
   (2,000 customers).
7. **Deployment**: Trained model + scaler served via a Flask REST API
   (`/predict` endpoint) returning churn probability and a risk tier.

---

## 📊 Results

| Metric          | Baseline (raw LogReg) | Final Model (tuned, engineered features) |
|-----------------|:---------------------:|:------------------------------------------:|
| Accuracy        | 0.825                  | 0.704                                       |
| Precision       | 0.710                  | 0.372                                       |
| **Recall**      | **0.170**              | **0.758**                                   |
| **F1-score**    | **0.274**              | **0.499**                                   |
| ROC-AUC         | 0.799                  | 0.799                                       |

**Key takeaway**: After feature engineering and class-weight balancing, the
model catches **75.8% of at-risk customers** (recall) — a **4.5x improvement**
over the baseline's 17% — while keeping ROC-AUC stable at **0.80**. This
trade-off (lower precision, much higher recall) is intentional and desirable
for churn use cases, where the cost of *missing* a churner (lost customer)
is typically much higher than the cost of a false alarm (a retention email
to a happy customer).

- **Best model**: Logistic Regression (`C=0.1`, `class_weight="balanced"`),
  selected from 3 candidates via 5-fold CV, beating Random Forest
  (CV AUC 0.722) and Gradient Boosting (CV AUC 0.738) with **CV AUC 0.753**.
- Confusion matrix, ROC curve, and feature importance plots are in `/plots`.

### Top predictive features
`is_premium_member`, `tenure_months`, `days_since_last_purchase`,
`email_open_rate`, and `total_purchases` were the strongest churn predictors
— aligning with intuition that loyalty status, recency, and engagement drive
retention.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate the dataset
python3 data/generate_data.py

# 3. Train the model (reproduces all results/plots above)
python3 src/train.py

# 4. Run inference on example customers
python3 src/predict_demo.py

# 5. Launch the REST API
python3 app/app.py
```

### Example API request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
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
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.953,
  "risk_level": "High"
}
```

---

## 🛠️ Tech Stack

- **Python 3.12**, **pandas**, **NumPy**
- **scikit-learn** — preprocessing, model training, cross-validation, GridSearchCV
- **Matplotlib** — evaluation visualizations
- **Flask** — REST API deployment
- **joblib** — model persistence

---

## 🔮 Future Improvements

- Add SHAP-based explainability for individual predictions
- Replace synthetic data with a real e-commerce churn dataset (e.g., Kaggle
  Telco Customer Churn) for benchmarking
- Add a model monitoring/retraining pipeline
- Containerize the Flask API with Docker
