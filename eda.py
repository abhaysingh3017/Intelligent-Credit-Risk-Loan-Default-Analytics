import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_preprocessing import load_data, clean_data, engineer_features

os.makedirs("static/plots", exist_ok=True)
PLOT_DIR = "static/plots"

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}.png", dpi=100, bbox_inches="tight")
    plt.close()

def plot_default_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["Loan_Default"].value_counts()
    axes[0].pie(counts, labels=["No Default", "Default"], autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"], startangle=90)
    axes[0].set_title("Loan Default Distribution")
    sns.countplot(x="Loan_Default", data=df, palette=["#2ecc71", "#e74c3c"], ax=axes[1])
    axes[1].set_title("Default Count")
    axes[1].set_xticklabels(["No Default", "Default"])
    save_fig("default_distribution")

def plot_credit_score_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x="Credit_Score", hue="Loan_Default", bins=40,
                 palette=["#2ecc71", "#e74c3c"], ax=axes[0], alpha=0.7)
    axes[0].set_title("Credit Score Distribution by Default Status")
    sns.boxplot(x="Loan_Default", y="Credit_Score", data=df,
                palette=["#2ecc71", "#e74c3c"], ax=axes[1])
    axes[1].set_xticklabels(["No Default", "Default"])
    axes[1].set_title("Credit Score vs Default")
    save_fig("credit_score_dist")

def plot_income_loan_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x="Monthly_Income", hue="Loan_Default", bins=40,
                 palette=["#2ecc71", "#e74c3c"], ax=axes[0], alpha=0.7)
    axes[0].set_title("Monthly Income Distribution by Default")
    sns.scatterplot(data=df.sample(2000, random_state=42),
                    x="Monthly_Income", y="Loan_Amount",
                    hue="Loan_Default", palette=["#2ecc71", "#e74c3c"],
                    alpha=0.5, ax=axes[1])
    axes[1].set_title("Income vs Loan Amount")
    save_fig("income_loan_analysis")

def plot_dti_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x="DTI_Ratio", hue="Loan_Default", bins=40,
                 palette=["#2ecc71", "#e74c3c"], ax=axes[0], alpha=0.7)
    axes[0].set_title("DTI Ratio Distribution by Default")
    axes[0].set_xlim(0, df["DTI_Ratio"].quantile(0.99))
    sns.boxplot(x="Loan_Default", y="Loan_to_Income", data=df,
                palette=["#2ecc71", "#e74c3c"], ax=axes[1])
    axes[1].set_xticklabels(["No Default", "Default"])
    axes[1].set_title("Loan-to-Income Ratio vs Default")
    save_fig("dti_analysis")

def plot_missed_payments(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    default_rate = df.groupby("Missed_Payments")["Loan_Default"].mean().reset_index()
    sns.barplot(x="Missed_Payments", y="Loan_Default", data=default_rate,
                palette="Reds", ax=axes[0])
    axes[0].set_title("Default Rate by Missed Payments")
    axes[0].set_ylabel("Default Rate")
    default_by_loans = df.groupby("Existing_Loans")["Loan_Default"].mean().reset_index()
    sns.barplot(x="Existing_Loans", y="Loan_Default", data=default_by_loans,
                palette="Blues", ax=axes[1])
    axes[1].set_title("Default Rate by Existing Loans")
    axes[1].set_ylabel("Default Rate")
    save_fig("missed_payments")

def plot_employment_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    emp_default = df.groupby("Employment_Label")["Loan_Default"].mean().reset_index()
    sns.barplot(x="Employment_Label", y="Loan_Default", data=emp_default,
                palette="Set2", ax=axes[0])
    axes[0].set_title("Default Rate by Employment Type")
    axes[0].set_ylabel("Default Rate")
    axes[0].tick_params(axis="x", rotation=15)
    age_default = df.groupby("Age_Group", observed=True)["Loan_Default"].mean().reset_index()
    sns.barplot(x="Age_Group", y="Loan_Default", data=age_default,
                palette="Set3", ax=axes[1])
    axes[1].set_title("Default Rate by Age Group")
    axes[1].set_ylabel("Default Rate")
    save_fig("employment_analysis")

def plot_correlation_heatmap(df):
    num_cols = ["Age", "Monthly_Income", "Loan_Amount", "Credit_Score",
                "Existing_Loans", "Missed_Payments", "DTI_Ratio",
                "Loan_to_Income", "Risk_Score", "Loan_Default"]
    corr = df[num_cols].corr()
    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    save_fig("correlation_heatmap")

def plot_credit_band_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    band_default = df.groupby("Credit_Band", observed=True)["Loan_Default"].mean().reset_index()
    sns.barplot(x="Credit_Band", y="Loan_Default", data=band_default,
                palette="RdYlGn", ax=axes[0])
    axes[0].set_title("Default Rate by Credit Band")
    axes[0].set_ylabel("Default Rate")
    axes[0].tick_params(axis="x", rotation=15)
    sns.countplot(x="Credit_Band", hue="Loan_Default", data=df,
                  palette=["#2ecc71", "#e74c3c"], ax=axes[1], order=band_default["Credit_Band"])
    axes[1].set_title("Count by Credit Band and Default")
    axes[1].tick_params(axis="x", rotation=15)
    save_fig("credit_band_analysis")

def generate_all_plots(df):
    print("Generating EDA plots...")
    plot_default_distribution(df)
    plot_credit_score_distribution(df)
    plot_income_loan_analysis(df)
    plot_dti_analysis(df)
    plot_missed_payments(df)
    plot_employment_analysis(df)
    plot_correlation_heatmap(df)
    plot_credit_band_analysis(df)
    print("All EDA plots saved to static/plots/")

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    generate_all_plots(df)
