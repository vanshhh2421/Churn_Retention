"""
generate_data.py
-----------------
Generates a synthetic but realistic e-commerce customer dataset for
churn prediction. Mimics common features found in retail/e-commerce
customer analytics (tenure, spend, support interactions, engagement).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000  # number of customers

# --- Core behavioral features -------------------------------------------------
tenure_months = np.random.gamma(shape=2.0, scale=8, size=N).clip(1, 72)
monthly_charges = np.random.normal(45, 18, size=N).clip(5, 150)
total_purchases = (np.random.poisson(lam=tenure_months / 2, size=N)).clip(0, 200)
avg_order_value = np.random.normal(38, 15, size=N).clip(5, 250)
num_support_tickets = np.random.poisson(lam=1.2, size=N).clip(0, 15)
days_since_last_purchase = np.random.exponential(scale=20, size=N).clip(0, 365)
is_premium_member = np.random.binomial(1, 0.35, size=N)
num_returns = np.random.poisson(lam=0.6, size=N).clip(0, 10)
app_sessions_per_week = np.random.gamma(shape=2.0, scale=2.0, size=N).clip(0, 30)
email_open_rate = np.random.beta(2, 3, size=N)  # between 0-1
discount_usage_rate = np.random.beta(2, 5, size=N)

# --- Construct churn probability using a logistic combination -----------------
z = (
    -0.05 * tenure_months
    + 0.015 * monthly_charges
    - 0.03 * total_purchases
    + 0.06 * num_support_tickets
    + 0.02 * days_since_last_purchase
    - 1.1 * is_premium_member
    + 0.25 * num_returns
    - 0.12 * app_sessions_per_week
    - 2.0 * email_open_rate
    + 0.5 * discount_usage_rate
    - 0.5  # intercept
)

prob_churn = 1 / (1 + np.exp(-z))
churn = np.random.binomial(1, prob_churn)

df = pd.DataFrame({
    "tenure_months": tenure_months.round(1),
    "monthly_charges": monthly_charges.round(2),
    "total_purchases": total_purchases,
    "avg_order_value": avg_order_value.round(2),
    "num_support_tickets": num_support_tickets,
    "days_since_last_purchase": days_since_last_purchase.round(1),
    "is_premium_member": is_premium_member,
    "num_returns": num_returns,
    "app_sessions_per_week": app_sessions_per_week.round(2),
    "email_open_rate": email_open_rate.round(3),
    "discount_usage_rate": discount_usage_rate.round(3),
    "churn": churn,
})

df.to_csv("/home/claude/churn_project/data/customer_data.csv", index=False)
print("Dataset created:", df.shape)
print("Churn rate: {:.2%}".format(df["churn"].mean()))
print(df.head())
