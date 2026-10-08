import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.data.preprocess import load_all_users, load_session
from src.features.extract_features import extract_features_from_session

TRAINING_DIR = "data/raw/Mouse-Dynamics-Challenge-master-2/training_files"
MY_DATA_DIR = "data/raw/my_data"

FEATURE_COLUMNS = [
    "total_distance", "avg_velocity", "max_velocity", "min_velocity",
    "avg_acceleration", "avg_jerk", "avg_direction_change",
    "straightness_ratio", "click_count", "click_frequency",
    "velocity_variance", "duration"
]

def compute_eer(y_test, y_scores):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2

def build_personal_dataset(window_size=50):
    rows = []

    for filename in os.listdir(MY_DATA_DIR):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(MY_DATA_DIR, filename)
        my_df = load_session(filepath)
        my_features = extract_features_from_session(my_df, window_size=window_size)
        my_features["label"] = 1
        my_features["user"] = "me"
        my_features["session"] = filename
        rows.append(my_features)
        print(f"{filename}: {len(my_features)} genuine windows")

    all_data = load_all_users(TRAINING_DIR)
    for user, sessions in all_data.items():
        for session_name, df in sessions.items():
            feats = extract_features_from_session(df, window_size=window_size)
            feats["label"] = 0
            feats["user"] = user
            feats["session"] = session_name
            rows.append(feats)

    dataset = pd.concat(rows, ignore_index=True)
    return dataset

if __name__ == "__main__":
    print("Building dataset with you as genuine (all sessions combined)...")
    dataset = build_personal_dataset(window_size=100)
    print("\nLabel distribution:")
    print(dataset["label"].value_counts())

    dataset.to_csv("data/processed/my_dataset.csv", index=False)

    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["impostor", "genuine"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    y_scores = model.predict_proba(X_test)[:, 1]
    eer = compute_eer(y_test, y_scores)
    print(f"EER: {eer:.4f}")

    joblib.dump(model, "src/models/saved/prerana_random_forest.pkl")
    print("Saved to src/models/saved/prerana_random_forest.pkl")