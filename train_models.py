import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from data_preprocessing import load_data, clean_data, engineer_features, get_feature_matrix

os.makedirs("static/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                                                   n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                       learning_rate=0.05, random_state=42)
}

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "ROC_AUC":   round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics, y_pred, y_prob, model

def plot_confusion_matrices(results_dict, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (name, (_, y_pred, _, _)) in enumerate(results_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["No Default", "Default"],
                    yticklabels=["No Default", "Default"])
        axes[i].set_title(f"{name}\nConfusion Matrix")
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("static/plots/confusion_matrices.png", dpi=100)
    plt.close()

def plot_roc_curves(results_dict, y_test):
    plt.figure(figsize=(10, 7))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for (name, (_, _, y_prob, _)), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("static/plots/roc_curves.png", dpi=100)
    plt.close()

def plot_metrics_comparison(metrics_list):
    df_m = pd.DataFrame(metrics_list).set_index("Model")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    df_m[metric_cols].plot(kind="bar", figsize=(14, 6), colormap="Set2", edgecolor="black")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("static/plots/model_comparison.png", dpi=100)
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    fi.plot(kind="barh", color="#3498db", edgecolor="black")
    plt.title(f"Feature Importance - {model_name}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("static/plots/feature_importance.png", dpi=100)
    plt.close()

def train_and_evaluate():
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    X, y = get_feature_matrix(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    # SMOTE on training set
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)

    results_dict = {}
    metrics_list = []

    for name, model in MODELS.items():
        print(f"Training {name}...")
        metrics, y_pred, y_prob, fitted_model = evaluate_model(
            name, model, X_train_res, X_test_s, y_train_res, y_test
        )
        results_dict[name] = (metrics, y_pred, y_prob, fitted_model)
        metrics_list.append(metrics)
        joblib.dump(fitted_model, f"models/{name.replace(' ', '_').lower()}.pkl")
        print(f"  {metrics}")

    # Save best model (by ROC-AUC)
    best = max(metrics_list, key=lambda x: x["ROC_AUC"])
    print(f"\nBest Model: {best['Model']} (ROC-AUC={best['ROC_AUC']})")
    best_model = results_dict[best["Model"]][3]
    joblib.dump(best_model, "models/best_model.pkl")

    # Save metrics JSON
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=2)

    # Save feature names
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Plots
    plot_confusion_matrices(results_dict, y_test)
    plot_roc_curves(results_dict, y_test)
    plot_metrics_comparison(metrics_list)
    plot_feature_importance(best_model, feature_names, best["Model"])

    # Classification report for best model
    y_pred_best = results_dict[best["Model"]][1]
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred_best, target_names=["No Default", "Default"]))

    return metrics_list, best["Model"]

if __name__ == "__main__":
    train_and_evaluate()
