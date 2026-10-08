from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from src.api.predict import predict_window
from src.api.trust_engine import TrustEngine
from src.api.drift_handler import DriftHandler
from src.features.extract_features import compute_window_features

app = Flask(__name__)
CORS(app)

trust_engines = {
    "user7": TrustEngine(window_history=5, decay=0.6),
    "prerana": TrustEngine(window_history=5, decay=0.6)
}
drift_handlers = {
    "user7": DriftHandler(base_threshold=0.5, history_size=20, adapt_rate=0.05),
    "prerana": DriftHandler(base_threshold=0.5, history_size=20, adapt_rate=0.05)
}

@app.route("/authenticate", methods=["POST"])
def authenticate():
    data = request.get_json()

    events = data.get("events")
    model_name = data.get("model", "prerana")

    if not events:
        return jsonify({"error": "No mouse events provided"}), 400

    window_df = pd.DataFrame(events)

    required_cols = {"x", "y", "client timestamp", "state"}
    if not required_cols.issubset(window_df.columns):
        return jsonify({"error": f"Missing required columns: {required_cols}"}), 400

    debug_feats = compute_window_features(window_df)
    print("DEBUG - RAW FEATURES FROM LIVE DATA:")
    for k, v in debug_feats.items():
        print(f"  {k}: {v}")

    try:
        result = predict_window(window_df, model_name=model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    trust_score = trust_engines[model_name].update(result["mouse_confidence"])
    trust_status = trust_engines[model_name].get_status(trust_score)

    drift_handlers[model_name].record(result["mouse_confidence"], result["decision"])
    current_threshold = drift_handlers[model_name].get_threshold()

    result["trust_score"] = trust_score
    result["trust_status"] = trust_status
    result["adaptive_threshold"] = current_threshold

    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)