import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve
from xgboost import XGBClassifier

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

def evaluate(name, y_test, y_pred, y_scores):
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["impostor", "genuine"]))
    eer = compute_eer(y_test, y_scores)
    print(f"EER: {eer:.4f}")
    return eer

def run_comparison(df):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_scores = rf.predict_proba(X_test)[:, 1]
    results["Random Forest"] = evaluate("Random Forest", y_test, y_pred, y_scores)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_scores = xgb.predict_proba(X_test)[:, 1]
    results["XGBoost"] = evaluate("XGBoost", y_test, y_pred, y_scores)

    scaler = StandardScaler()
    X_train_genuine = X_train[y_train == 1]
    X_train_genuine_scaled = scaler.fit_transform(X_train_genuine)
    X_test_scaled = scaler.transform(X_test)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    ocsvm.fit(X_train_genuine_scaled)
    raw_pred = ocsvm.predict(X_test_scaled)
    y_pred = np.where(raw_pred == 1, 1, 0)
    y_scores = ocsvm.decision_function(X_test_scaled)
    results["One-Class SVM"] = evaluate("One-Class SVM", y_test, y_pred, y_scores)

    iso = IsolationForest(n_estimators=300, contamination=0.1, random_state=42)
    iso.fit(X_train_genuine_scaled)
    raw_pred = iso.predict(X_test_scaled)
    y_pred = np.where(raw_pred == 1, 1, 0)
    y_scores = iso.decision_function(X_test_scaled)
    results["Isolation Forest"] = evaluate("Isolation Forest", y_test, y_pred, y_scores)

    print("\n===== SUMMARY: EER per model (lower is better) =====")
    for name, eer in results.items():
        print(f"{name}: {eer:.4f}")

if __name__ == "__main__":
    df = pd.read_csv("data/processed/user7_dataset.csv")
    run_comparison(df)