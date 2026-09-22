import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
from data_preprocessing import load_data, clean_data, engineer_features

os.makedirs("static/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

def run_clustering(df, n_clusters=4):
    cluster_features = [
        "Age", "Monthly_Income", "Loan_Amount",
        "Credit_Score", "DTI_Ratio", "Missed_Payments",
        "Existing_Loans", "Risk_Score"
    ]
    X = df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    inertias = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(K_range), inertias, "bo-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.tight_layout()
    plt.savefig("static/plots/elbow_curve.png", dpi=100)
    plt.close()

    # Final model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    joblib.dump(kmeans, "models/kmeans.pkl")
    joblib.dump(scaler, "models/cluster_scaler.pkl")

    # PCA visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    plt.figure(figsize=(10, 7))
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for c in range(n_clusters):
        mask = df["Cluster"] == c
        plt.scatter(df.loc[mask, "PCA1"], df.loc[mask, "PCA2"],
                    label=f"Cluster {c}", alpha=0.5, s=10, color=palette[c])
    plt.title("Customer Segments (PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/plots/cluster_pca.png", dpi=100)
    plt.close()

    # Cluster profiles
    profile_cols = ["Age", "Monthly_Income", "Loan_Amount", "Credit_Score",
                    "DTI_Ratio", "Missed_Payments", "Existing_Loans", "Loan_Default"]
    cluster_profile = df.groupby("Cluster")[profile_cols].mean().round(2)
    cluster_profile["Count"] = df.groupby("Cluster").size()
    cluster_profile["Default_Rate_%"] = (cluster_profile["Loan_Default"] * 100).round(1)

    # Cluster heatmap
    plt.figure(figsize=(12, 5))
    norm_profile = (cluster_profile[profile_cols[:-1]] - cluster_profile[profile_cols[:-1]].min()) / \
                   (cluster_profile[profile_cols[:-1]].max() - cluster_profile[profile_cols[:-1]].min())
    sns.heatmap(norm_profile, annot=cluster_profile[profile_cols[:-1]].values,
                fmt=".1f", cmap="YlOrRd", linewidths=0.5)
    plt.title("Cluster Profile Heatmap (Normalized)")
    plt.tight_layout()
    plt.savefig("static/plots/cluster_heatmap.png", dpi=100)
    plt.close()

    print("\nCluster Profiles:")
    print(cluster_profile.to_string())
    return df, cluster_profile

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df_clustered, profile = run_clustering(df)
    print("\nDone. Plots saved.")
