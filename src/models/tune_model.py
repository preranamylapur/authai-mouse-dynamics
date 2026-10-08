import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import numpy as np
import joblib

FEATURE_COLUMNS = [
    "total_distance", "avg_velocity", "max_velocity", "min_velocity",
    "avg_acceleration", "avg_jerk", "avg_direction_change",
    "straightness_ratio", "click_count", "click_frequency",
    "velocity_variance", "duration"
]

def compute_far_frr_eer(y_test, y_scores):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    print(f"FAR: {fpr[eer_index]:.4f}  FRR: {fnr[eer_index]:.4f}  EER: {eer:.4f}")
    return eer

def tune_model(df):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV F1 score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=["impostor", "genuine"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    y_scores = best_model.predict_proba(X_test)[:, 1]
    compute_far_frr_eer(y_test, y_scores)

    return best_model

if __name__ == "__main__":
    df = pd.read_csv("data/processed/user7_dataset.csv")
    best_model = tune_model(df)

    joblib.dump(best_model, "src/models/saved/user7_random_forest_tuned.pkl")
    print("Saved tuned model to src/models/saved/user7_random_forest_tuned.pkl")