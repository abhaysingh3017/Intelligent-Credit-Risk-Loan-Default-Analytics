import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

DATA_PATH = "Financial Risk Modeling P.csv"

EMPLOYMENT_MAP = {0: "Self-Employed", 1: "Salaried", 2: "Business", 3: "Freelancer"}
CITY_MAP = {
    0: "Mumbai", 1: "Delhi", 2: "Bangalore", 3: "Chennai",
    4: "Hyderabad", 5: "Pune"
}

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def clean_data(df):
    df = df.copy()
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Clip outliers: Monthly_Income floor at 10000 (already present), Credit_Score 300-850
    df["Monthly_Income"] = df["Monthly_Income"].clip(lower=10000)
    df["Credit_Score"] = df["Credit_Score"].clip(300, 850)
    df["Existing_Loans"] = df["Existing_Loans"].clip(lower=0)
    df["Missed_Payments"] = df["Missed_Payments"].clip(lower=0)
    return df

def engineer_features(df):
    df = df.copy()
    # Debt-to-Income Ratio (monthly loan payment proxy = loan/60 months)
    df["Monthly_Loan_Payment"] = df["Loan_Amount"] / 60
    df["DTI_Ratio"] = df["Monthly_Loan_Payment"] / df["Monthly_Income"]
    # Loan-to-Income Ratio
    df["Loan_to_Income"] = df["Loan_Amount"] / df["Monthly_Income"]
    # Credit Score Band
    df["Credit_Band"] = pd.cut(
        df["Credit_Score"],
        bins=[0, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Exceptional"]
    )
    # Risk Score (composite)
    df["Risk_Score"] = (
        (df["Missed_Payments"] * 2) +
        (df["Existing_Loans"]) +
        (df["DTI_Ratio"] * 10) -
        (df["Credit_Score"] / 100)
    )
    # Age Group
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[18, 30, 40, 50, 60, 70],
        labels=["18-30", "31-40", "41-50", "51-60", "61-70"]
    )
    # Employment label
    df["Employment_Label"] = df["Employment_Type"].map(EMPLOYMENT_MAP)
    return df

def get_feature_matrix(df):
    features = [
        "Age", "Employment_Type", "Monthly_Income", "Loan_Amount",
        "Credit_Score", "Existing_Loans", "Missed_Payments",
        "DTI_Ratio", "Loan_to_Income", "Risk_Score"
    ]
    X = df[features]
    y = df["Loan_Default"]
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    print("Shape:", df.shape)
    print(df[["DTI_Ratio", "Loan_to_Income", "Risk_Score", "Credit_Band"]].head())
    print("Default Rate:", df["Loan_Default"].mean().round(4))
