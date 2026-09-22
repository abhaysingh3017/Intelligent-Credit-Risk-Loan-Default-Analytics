"""
setup.py - Run this ONCE to train all models and generate all plots.
After setup completes, run app.py to start the Flask dashboard.
"""
import os
import sys

print("=" * 60)
print("  Credit Risk Analytics System - Setup")
print("=" * 60)

print("\n[1/3] Running EDA and generating plots...")
from data_preprocessing import load_data, clean_data, engineer_features
from eda import generate_all_plots

df = load_data()
df = clean_data(df)
df = engineer_features(df)
generate_all_plots(df)
print("  EDA plots done.")

print("\n[2/3] Running customer segmentation (K-Means)...")
from clustering import run_clustering
_, profile = run_clustering(df)
print("  Clustering done.")

print("\n[3/3] Training ML models (this may take 1-2 minutes)...")
from train_models import train_and_evaluate
metrics, best_name = train_and_evaluate()
print(f"  Training done. Best model: {best_name}")

print("\n" + "=" * 60)
print("  Setup complete!")
print("  Run:  python app.py")
print("  Then open: http://127.0.0.1:5000")
print("=" * 60)
