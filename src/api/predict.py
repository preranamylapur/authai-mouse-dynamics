import joblib
import pandas as pd
from src.features.extract_features import compute_window_features

FEATURE_COLUMNS = [
    "total_distance", "avg_velocity", "max_velocity", "min_velocity",
    "avg_acceleration", "avg_jerk", "avg_direction_change",
    "straightness_ratio", "click_count", "click_frequency",
    "velocity_variance", "duration"
]

MODEL_PATHS = {
    "user7": "src/models/saved/user7_random_forest_tuned.pkl",
    "prerana": "src/models/saved/prerana_random_forest.pkl"
}

_loaded_models = {}

def get_model(model_name):
    if model_name not in _loaded_models:
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Unknown model: {model_name}")
        _loaded_models[model_name] = joblib.load(MODEL_PATHS[model_name])
    return _loaded_models[model_name]

def predict_window(window_df, model_name="prerana"):
    model = get_model(model_name)
    feats = compute_window_features(window_df)
    feats_df = pd.DataFrame([feats])[FEATURE_COLUMNS]

    confidence = model.predict_proba(feats_df)[0][1]
    risk = 1 - confidence
    threshold = 0.3 if model_name == "prerana" else 0.5
    decision = "genuine" if confidence >= threshold else "impostor"
    return {
        "mouse_confidence": round(float(confidence), 4),
        "mouse_risk": round(float(risk), 4),
        "decision": decision,
        "active_model": model_name
    }

if __name__ == "__main__":
    from src.data.preprocess import load_session

    df = load_session("data/raw/Mouse-Dynamics-Challenge-master-2/training_files/user7/session_1060325796")
    window = df.iloc[0:100]

    result = predict_window(window, model_name="user7")
    print(result)