import clustering
import data_preprocessing
import train_models

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os
from data_preprocessing import load_data, clean_data, engineer_features

app = Flask(__name__)

# Load artifacts

scaler      = joblib.load("models/scaler.pkl")
best_model  = joblib.load("models/best_model.pkl")
with open("models/metrics.json") as f:
    metrics_list = json.load(f)
with open("models/feature_names.json") as f:
    feature_names = json.load(f)

def get_summary_stats():
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    stats = {
        "total_customers":  int(len(df)),
        "default_rate":     round(float(df["Loan_Default"].mean()) * 100, 2),
        "avg_credit_score": round(float(df["Credit_Score"].mean()), 1),
        "avg_loan_amount":  round(float(df["Loan_Amount"].mean()), 0),
        "avg_dti":          round(float(df["DTI_Ratio"].mean()), 4),
        "avg_income":       round(float(df["Monthly_Income"].mean()), 0),
    }
    return stats

@app.route("/")
def index():
    stats = get_summary_stats()
    best = max(metrics_list, key=lambda x: x["ROC_AUC"])
    return render_template("index.html", stats=stats, metrics=metrics_list, best=best)

@app.route("/eda")
def eda():
    return render_template("eda.html")

@app.route("/models")
def models_page():
    best = max(metrics_list, key=lambda x: x["ROC_AUC"])
    return render_template("models.html", metrics=metrics_list, best=best)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            age             = float(request.form["age"])
            employment_type = int(request.form["employment_type"])
            monthly_income  = float(request.form["monthly_income"])
            loan_amount     = float(request.form["loan_amount"])
            credit_score    = float(request.form["credit_score"])
            existing_loans  = int(request.form["existing_loans"])
            missed_payments = int(request.form["missed_payments"])

            # Validate: reject NaN/Inf and out-of-range values
            inputs = [age, monthly_income, loan_amount, credit_score]
            if any(not np.isfinite(v) for v in inputs):
                raise ValueError("Input contains NaN or Inf values.")
            if not (18 <= age <= 100):
                raise ValueError("Age must be between 18 and 100.")
            if monthly_income <= 0:
                raise ValueError("Monthly income must be positive.")
            if loan_amount <= 0:
                raise ValueError("Loan amount must be positive.")
            if not (300 <= credit_score <= 850):
                raise ValueError("Credit score must be between 300 and 850.")
            if existing_loans < 0 or missed_payments < 0:
                raise ValueError("Loans and missed payments cannot be negative.")

            monthly_loan_payment = loan_amount / 60
            dti_ratio            = monthly_loan_payment / monthly_income
            loan_to_income       = loan_amount / monthly_income
            risk_score           = (missed_payments * 2) + existing_loans + (dti_ratio * 10) - (credit_score / 100)

            features = np.array([[
                age, employment_type, monthly_income, loan_amount,
                credit_score, existing_loans, missed_payments,
                dti_ratio, loan_to_income, risk_score
            ]])
            features_scaled = scaler.transform(features)
            prediction      = best_model.predict(features_scaled)[0]
            probability     = best_model.predict_proba(features_scaled)[0][1]

            risk_level = "Low" if probability < 0.4 else ("Medium" if probability < 0.65 else "High")
            result = {
                "prediction":  int(prediction),
                "probability": round(float(probability) * 100, 2),
                "risk_level":  risk_level,
                "dti_ratio":   round(dti_ratio, 4),
                "loan_to_income": round(loan_to_income, 2),
                "risk_score":  round(risk_score, 2),
            }
        except Exception as e:
            result = {"error": str(e)}
    return render_template("predict.html", result=result)

@app.route("/api/metrics")
def api_metrics():
    return jsonify(metrics_list)

@app.route("/api/stats")
def api_stats():
    return jsonify(get_summary_stats())

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=5000)
