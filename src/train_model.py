"""
train_model.py
---------------
Trains a DecisionTreeClassifier on data/placement_data.csv, evaluates it,
saves the trained model with joblib, and saves two supporting charts:
  - charts/feature_importance.png
  - charts/dataset_overview.png

Run:
    python src/train_model.py
"""

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/placement_data.csv"
MODEL_PATH = "model/placement_model.pkl"
CHART_DIR = "charts"

FEATURES = [
    "cgpa",
    "backlogs",
    "internships",
    "projects",
    "coding_score",
    "communication_score",
    "attendance_percent",
]
TARGET = "placed"


def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=15,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    joblib.dump({"model": clf, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # --- Chart 1: Feature importance ---
    importances = clf.feature_importances_
    order = importances.argsort()[::-1]
    plt.figure(figsize=(7, 4))
    plt.barh([FEATURES[i] for i in order][::-1], importances[order][::-1], color="#1f4e79")
    plt.title("Feature Importance — Placement Prediction Model")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/feature_importance.png", dpi=150)
    plt.close()

    # --- Chart 2: Dataset overview (CGPA distribution by placement outcome) ---
    plt.figure(figsize=(7, 4))
    df[df.placed == 1]["cgpa"].hist(alpha=0.6, bins=20, label="Placed", color="#2e7d32")
    df[df.placed == 0]["cgpa"].hist(alpha=0.6, bins=20, label="Not placed", color="#c62828")
    plt.title("Dataset Overview — CGPA Distribution by Outcome")
    plt.xlabel("CGPA")
    plt.ylabel("Number of students")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/dataset_overview.png", dpi=150)
    plt.close()

    print(f"Saved charts to {CHART_DIR}/")


if __name__ == "__main__":
    main()
