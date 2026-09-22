#!/usr/bin/env python
import contextlib as __stickytape_contextlib

@__stickytape_contextlib.contextmanager
def __stickytape_temporary_dir():
    import tempfile
    import shutil
    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)

with __stickytape_temporary_dir() as __stickytape_working_dir:
    def __stickytape_write_module(path, contents):
        import os, os.path

        def make_package(path):
            parts = path.split("/")
            partial_path = __stickytape_working_dir
            for part in parts:
                partial_path = os.path.join(partial_path, part)
                if not os.path.exists(partial_path):
                    os.mkdir(partial_path)
                    with open(os.path.join(partial_path, "__init__.py"), "wb") as f:
                        f.write(b"\n")

        make_package(os.path.dirname(path))

        full_path = os.path.join(__stickytape_working_dir, path)
        with open(full_path, "wb") as module_file:
            module_file.write(contents)

    import sys as __stickytape_sys
    __stickytape_sys.path.insert(0, __stickytape_working_dir)

    __stickytape_write_module('clustering.py', b'import pandas as pd\nimport numpy as np\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nimport joblib\nimport os\nfrom data_preprocessing import load_data, clean_data, engineer_features\n\nos.makedirs("static/plots", exist_ok=True)\nos.makedirs("models", exist_ok=True)\n\ndef run_clustering(df, n_clusters=4):\n    cluster_features = [\n        "Age", "Monthly_Income", "Loan_Amount",\n        "Credit_Score", "DTI_Ratio", "Missed_Payments",\n        "Existing_Loans", "Risk_Score"\n    ]\n    X = df[cluster_features].copy()\n    scaler = StandardScaler()\n    X_scaled = scaler.fit_transform(X)\n\n    # Elbow method\n    inertias = []\n    K_range = range(2, 9)\n    for k in K_range:\n        km = KMeans(n_clusters=k, random_state=42, n_init=10)\n        km.fit(X_scaled)\n        inertias.append(km.inertia_)\n\n    plt.figure(figsize=(8, 4))\n    plt.plot(list(K_range), inertias, "bo-")\n    plt.xlabel("Number of Clusters")\n    plt.ylabel("Inertia")\n    plt.title("Elbow Method for Optimal K")\n    plt.tight_layout()\n    plt.savefig("static/plots/elbow_curve.png", dpi=100)\n    plt.close()\n\n    # Final model\n    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)\n    df = df.copy()\n    df["Cluster"] = kmeans.fit_predict(X_scaled)\n\n    joblib.dump(kmeans, "models/kmeans.pkl")\n    joblib.dump(scaler, "models/cluster_scaler.pkl")\n\n    # PCA visualization\n    pca = PCA(n_components=2, random_state=42)\n    X_pca = pca.fit_transform(X_scaled)\n    df["PCA1"] = X_pca[:, 0]\n    df["PCA2"] = X_pca[:, 1]\n\n    plt.figure(figsize=(10, 7))\n    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]\n    for c in range(n_clusters):\n        mask = df["Cluster"] == c\n        plt.scatter(df.loc[mask, "PCA1"], df.loc[mask, "PCA2"],\n                    label=f"Cluster {c}", alpha=0.5, s=10, color=palette[c])\n    plt.title("Customer Segments (PCA Projection)")\n    plt.xlabel("PCA Component 1")\n    plt.ylabel("PCA Component 2")\n    plt.legend()\n    plt.tight_layout()\n    plt.savefig("static/plots/cluster_pca.png", dpi=100)\n    plt.close()\n\n    # Cluster profiles\n    profile_cols = ["Age", "Monthly_Income", "Loan_Amount", "Credit_Score",\n                    "DTI_Ratio", "Missed_Payments", "Existing_Loans", "Loan_Default"]\n    cluster_profile = df.groupby("Cluster")[profile_cols].mean().round(2)\n    cluster_profile["Count"] = df.groupby("Cluster").size()\n    cluster_profile["Default_Rate_%"] = (cluster_profile["Loan_Default"] * 100).round(1)\n\n    # Cluster heatmap\n    plt.figure(figsize=(12, 5))\n    norm_profile = (cluster_profile[profile_cols[:-1]] - cluster_profile[profile_cols[:-1]].min()) / \\\n                   (cluster_profile[profile_cols[:-1]].max() - cluster_profile[profile_cols[:-1]].min())\n    sns.heatmap(norm_profile, annot=cluster_profile[profile_cols[:-1]].values,\n                fmt=".1f", cmap="YlOrRd", linewidths=0.5)\n    plt.title("Cluster Profile Heatmap (Normalized)")\n    plt.tight_layout()\n    plt.savefig("static/plots/cluster_heatmap.png", dpi=100)\n    plt.close()\n\n    print("\\nCluster Profiles:")\n    print(cluster_profile.to_string())\n    return df, cluster_profile\n\nif __name__ == "__main__":\n    df = load_data()\n    df = clean_data(df)\n    df = engineer_features(df)\n    df_clustered, profile = run_clustering(df)\n    print("\\nDone. Plots saved.")\n')
    __stickytape_write_module('data_preprocessing.py', b'import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nimport joblib\nimport os\n\nDATA_PATH = "Financial Risk Modeling P.csv"\n\nEMPLOYMENT_MAP = {0: "Self-Employed", 1: "Salaried", 2: "Business", 3: "Freelancer"}\nCITY_MAP = {\n    0: "Mumbai", 1: "Delhi", 2: "Bangalore", 3: "Chennai",\n    4: "Hyderabad", 5: "Pune"\n}\n\ndef load_data():\n    df = pd.read_csv(DATA_PATH)\n    return df\n\ndef clean_data(df):\n    df = df.copy()\n    # Drop duplicates\n    df.drop_duplicates(inplace=True)\n    # Clip outliers: Monthly_Income floor at 10000 (already present), Credit_Score 300-850\n    df["Monthly_Income"] = df["Monthly_Income"].clip(lower=10000)\n    df["Credit_Score"] = df["Credit_Score"].clip(300, 850)\n    df["Existing_Loans"] = df["Existing_Loans"].clip(lower=0)\n    df["Missed_Payments"] = df["Missed_Payments"].clip(lower=0)\n    return df\n\ndef engineer_features(df):\n    df = df.copy()\n    # Debt-to-Income Ratio (monthly loan payment proxy = loan/60 months)\n    df["Monthly_Loan_Payment"] = df["Loan_Amount"] / 60\n    df["DTI_Ratio"] = df["Monthly_Loan_Payment"] / df["Monthly_Income"]\n    # Loan-to-Income Ratio\n    df["Loan_to_Income"] = df["Loan_Amount"] / df["Monthly_Income"]\n    # Credit Score Band\n    df["Credit_Band"] = pd.cut(\n        df["Credit_Score"],\n        bins=[0, 580, 670, 740, 800, 850],\n        labels=["Poor", "Fair", "Good", "Very Good", "Exceptional"]\n    )\n    # Risk Score (composite)\n    df["Risk_Score"] = (\n        (df["Missed_Payments"] * 2) +\n        (df["Existing_Loans"]) +\n        (df["DTI_Ratio"] * 10) -\n        (df["Credit_Score"] / 100)\n    )\n    # Age Group\n    df["Age_Group"] = pd.cut(\n        df["Age"],\n        bins=[18, 30, 40, 50, 60, 70],\n        labels=["18-30", "31-40", "41-50", "51-60", "61-70"]\n    )\n    # Employment label\n    df["Employment_Label"] = df["Employment_Type"].map(EMPLOYMENT_MAP)\n    return df\n\ndef get_feature_matrix(df):\n    features = [\n        "Age", "Employment_Type", "Monthly_Income", "Loan_Amount",\n        "Credit_Score", "Existing_Loans", "Missed_Payments",\n        "DTI_Ratio", "Loan_to_Income", "Risk_Score"\n    ]\n    X = df[features]\n    y = df["Loan_Default"]\n    return X, y\n\ndef scale_features(X_train, X_test):\n    scaler = StandardScaler()\n    X_train_scaled = scaler.fit_transform(X_train)\n    X_test_scaled = scaler.transform(X_test)\n    os.makedirs("models", exist_ok=True)\n    joblib.dump(scaler, "models/scaler.pkl")\n    return X_train_scaled, X_test_scaled, scaler\n\nif __name__ == "__main__":\n    df = load_data()\n    df = clean_data(df)\n    df = engineer_features(df)\n    print("Shape:", df.shape)\n    print(df[["DTI_Ratio", "Loan_to_Income", "Risk_Score", "Credit_Band"]].head())\n    print("Default Rate:", df["Loan_Default"].mean().round(4))\n')
    __stickytape_write_module('train_models.py', b'import pandas as pd\nimport numpy as np\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport joblib\nimport os\nimport json\n\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.metrics import (\n    accuracy_score, precision_score, recall_score, f1_score,\n    roc_auc_score, confusion_matrix, roc_curve, classification_report\n)\nfrom sklearn.preprocessing import StandardScaler\nfrom imblearn.over_sampling import SMOTE\n\nfrom data_preprocessing import load_data, clean_data, engineer_features, get_feature_matrix\n\nos.makedirs("static/plots", exist_ok=True)\nos.makedirs("models", exist_ok=True)\n\nMODELS = {\n    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),\n    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),\n    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,\n                                                   n_jobs=-1),\n    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=5,\n                                                       learning_rate=0.05, random_state=42)\n}\n\ndef evaluate_model(name, model, X_train, X_test, y_train, y_test):\n    model.fit(X_train, y_train)\n    y_pred = model.predict(X_test)\n    y_prob = model.predict_proba(X_test)[:, 1]\n    metrics = {\n        "Model":     name,\n        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),\n        "Precision": round(precision_score(y_test, y_pred), 4),\n        "Recall":    round(recall_score(y_test, y_pred), 4),\n        "F1":        round(f1_score(y_test, y_pred), 4),\n        "ROC_AUC":   round(roc_auc_score(y_test, y_prob), 4),\n    }\n    return metrics, y_pred, y_prob, model\n\ndef plot_confusion_matrices(results_dict, y_test):\n    fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n    axes = axes.flatten()\n    for i, (name, (_, y_pred, _, _)) in enumerate(results_dict.items()):\n        cm = confusion_matrix(y_test, y_pred)\n        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],\n                    xticklabels=["No Default", "Default"],\n                    yticklabels=["No Default", "Default"])\n        axes[i].set_title(f"{name}\\nConfusion Matrix")\n        axes[i].set_ylabel("Actual")\n        axes[i].set_xlabel("Predicted")\n    plt.tight_layout()\n    plt.savefig("static/plots/confusion_matrices.png", dpi=100)\n    plt.close()\n\ndef plot_roc_curves(results_dict, y_test):\n    plt.figure(figsize=(10, 7))\n    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]\n    for (name, (_, _, y_prob, _)), color in zip(results_dict.items(), colors):\n        fpr, tpr, _ = roc_curve(y_test, y_prob)\n        auc = roc_auc_score(y_test, y_prob)\n        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")\n    plt.plot([0, 1], [0, 1], "k--", lw=1)\n    plt.xlabel("False Positive Rate")\n    plt.ylabel("True Positive Rate")\n    plt.title("ROC Curves - All Models")\n    plt.legend(loc="lower right")\n    plt.tight_layout()\n    plt.savefig("static/plots/roc_curves.png", dpi=100)\n    plt.close()\n\ndef plot_metrics_comparison(metrics_list):\n    df_m = pd.DataFrame(metrics_list).set_index("Model")\n    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]\n    df_m[metric_cols].plot(kind="bar", figsize=(14, 6), colormap="Set2", edgecolor="black")\n    plt.title("Model Performance Comparison")\n    plt.ylabel("Score")\n    plt.xticks(rotation=15)\n    plt.ylim(0, 1.05)\n    plt.legend(loc="lower right")\n    plt.tight_layout()\n    plt.savefig("static/plots/model_comparison.png", dpi=100)\n    plt.close()\n\ndef plot_feature_importance(model, feature_names, model_name):\n    if hasattr(model, "feature_importances_"):\n        importances = model.feature_importances_\n    elif hasattr(model, "coef_"):\n        importances = np.abs(model.coef_[0])\n    else:\n        return\n    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)\n    plt.figure(figsize=(10, 6))\n    fi.plot(kind="barh", color="#3498db", edgecolor="black")\n    plt.title(f"Feature Importance - {model_name}")\n    plt.xlabel("Importance Score")\n    plt.tight_layout()\n    plt.savefig("static/plots/feature_importance.png", dpi=100)\n    plt.close()\n\ndef train_and_evaluate():\n    df = load_data()\n    df = clean_data(df)\n    df = engineer_features(df)\n    X, y = get_feature_matrix(df)\n    feature_names = X.columns.tolist()\n\n    X_train, X_test, y_train, y_test = train_test_split(\n        X, y, test_size=0.2, random_state=42, stratify=y\n    )\n\n    # Scale\n    scaler = StandardScaler()\n    X_train_s = scaler.fit_transform(X_train)\n    X_test_s  = scaler.transform(X_test)\n    joblib.dump(scaler, "models/scaler.pkl")\n\n    # SMOTE on training set\n    sm = SMOTE(random_state=42)\n    X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)\n\n    results_dict = {}\n    metrics_list = []\n\n    for name, model in MODELS.items():\n        print(f"Training {name}...")\n        metrics, y_pred, y_prob, fitted_model = evaluate_model(\n            name, model, X_train_res, X_test_s, y_train_res, y_test\n        )\n        results_dict[name] = (metrics, y_pred, y_prob, fitted_model)\n        metrics_list.append(metrics)\n        joblib.dump(fitted_model, f"models/{name.replace(\' \', \'_\').lower()}.pkl")\n        print(f"  {metrics}")\n\n    # Save best model (by ROC-AUC)\n    best = max(metrics_list, key=lambda x: x["ROC_AUC"])\n    print(f"\\nBest Model: {best[\'Model\']} (ROC-AUC={best[\'ROC_AUC\']})")\n    best_model = results_dict[best["Model"]][3]\n    joblib.dump(best_model, "models/best_model.pkl")\n\n    # Save metrics JSON\n    with open("models/metrics.json", "w") as f:\n        json.dump(metrics_list, f, indent=2)\n\n    # Save feature names\n    with open("models/feature_names.json", "w") as f:\n        json.dump(feature_names, f)\n\n    # Plots\n    plot_confusion_matrices(results_dict, y_test)\n    plot_roc_curves(results_dict, y_test)\n    plot_metrics_comparison(metrics_list)\n    plot_feature_importance(best_model, feature_names, best["Model"])\n\n    # Classification report for best model\n    y_pred_best = results_dict[best["Model"]][1]\n    print("\\nClassification Report (Best Model):")\n    print(classification_report(y_test, y_pred_best, target_names=["No Default", "Default"]))\n\n    return metrics_list, best["Model"]\n\nif __name__ == "__main__":\n    train_and_evaluate()\n')
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
    