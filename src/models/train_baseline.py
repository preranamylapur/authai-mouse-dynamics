import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLUMNS = [
    "total_distance", "avg_velocity", "max_velocity", "min_velocity",
    "avg_acceleration", "avg_jerk", "avg_direction_change",
    "straightness_ratio", "click_count", "click_frequency",
    "velocity_variance", "duration"
]

def load_dataset(path):
    return pd.read_csv(path)

def train_and_evaluate(df):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=["impostor", "genuine"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model

import numpy as np
from sklearn.metrics import roc_curve

def compute_far_frr_eer(y_test, y_scores):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    fnr = 1 - tpr

    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]

    print(f"FAR at EER point: {fpr[eer_index]:.4f}")
    print(f"FRR at EER point: {fnr[eer_index]:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"Threshold at EER: {eer_threshold:.4f}")

    return eer, eer_threshold

import numpy as np
from sklearn.metrics import roc_curve

def compute_far_frr_eer(y_test, y_scores):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    fnr = 1 - tpr

    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]

    print(f"FAR at EER point: {fpr[eer_index]:.4f}")
    print(f"FRR at EER point: {fnr[eer_index]:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"Threshold at EER: {eer_threshold:.4f}")

    return eer, eer_threshold

if __name__ == "__main__":
    df = load_dataset("data/processed/user7_dataset.csv")
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["impostor", "genuine"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    y_scores = model.predict_proba(X_test)[:, 1]
    compute_far_frr_eer(y_test, y_scores)
    import joblib
    import os
    os.makedirs("src/models/saved", exist_ok=True)
    joblib.dump(model, "src/models/saved/user7_random_forest.pkl")
    print("Model saved to src/models/saved/user7_random_forest.pkl")