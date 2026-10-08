# AuthAI — Mouse Dynamics Authentication Module

> ## ➡️ START HERE: read [`MOUSE_HANDOVER.md`](MOUSE_HANDOVER.md)
> The **current, final version** is the **physical-mouse module** (`src/models/live_test.py`, port **5050**, endpoint `POST /authenticate`).
> Setup, running, the integration API, fusion output and final results are all in `MOUSE_HANDOVER.md`.
>
> Everything below this line documents the **earlier trackpad version** (`src/api/app.py`, port 5000). It is kept as the **baseline experiment** and is **not** the version to integrate.
> Note: the Balabit-only results (user7, EER ≈ 0.08) are valid within that dataset. The **personal trackpad model** (EER 0.02–0.03) compared the user against Balabit users, who were recorded with a different setup. The physical-mouse experiments showed that this setup difference, not the person, drives that score (see `MOUSE_HANDOVER.md` §8).

---

# (Baseline) Trackpad version — original README

**Author:** Prerana C M
**Module:** Mouse Dynamics Continuous Authentication
**Part of:** AuthAI — Continuous Identity Verification in a Zero-Trust World

## Status: Functional, tested, ready for multimodal integration discussion

---

## What This Module Does
Authenticates a user in real time based on how they move their mouse (behavioral biometrics), independent of passwords. Outputs a confidence score, decision (genuine/impostor), trust score, and adaptive threshold — designed to plug into a larger multimodal trust fusion engine alongside face and keystroke modules.

## Architecture

Raw mouse events (browser)
    -> Flask REST API (/authenticate)
    -> Feature extraction (12 behavioral features)
    -> Trained Random Forest model
    -> Trust engine (rolling smoothed score)
    -> Drift handler (adaptive threshold)
    -> JSON response

## Results

### Model trained on Balabit Mouse Dynamics Challenge (user7 as genuine)
| Metric | Value |
|---|---|
| Accuracy | 92% |
| Precision (genuine) | 0.72-0.75 |
| Recall (genuine) | 0.88-0.91 |
| EER | 0.081-0.085 |

Compared against original baseline report: Accuracy 74.92%, Precision 23.24%, EER 0.3366 — this is a ~75% reduction in EER.

### Model comparison (all trained/evaluated identically)
| Model | EER |
|---|---|
| XGBoost | 0.0811 (best) |
| Random Forest | 0.0846 |
| Isolation Forest | 0.1949 |
| One-Class SVM | 0.2049 |

### Personal model (Prerana as genuine, live-enrolled)
| Metric | Value |
|---|---|
| Precision (genuine) | 0.74 |
| Recall (genuine) | 0.83-0.93 |
| EER | 0.02-0.03 |

Live-tested: correctly authenticates the enrolled user (~0.95+ confidence) and correctly rejects a different movement pattern (0.0 confidence, tested with non-dominant hand as impostor proxy).
## API Contract (for teammate integration)

**POST** `/authenticate`

Request:
{
  "events": [{"x": 100, "y": 200, "client timestamp": 0.5, "state": "Move"}, ...],
  "model": "prerana"
}

Response:
{
  "mouse_confidence": 0.97,
  "mouse_risk": 0.03,
  "decision": "genuine",
  "active_model": "prerana",
  "trust_score": 0.94,
  "trust_status": "trusted",
  "adaptive_threshold": 0.5
}

**GET** `/health` -> `{"status": "ok"}`

## Multimodal Fusion (Phase 13) — not implemented in this folder
*(Note: `fusion_engine.py` was planned but is **not included**; for fusion use the output described in `MOUSE_HANDOVER.md` §7.)* The planned design was `compute_final_trust_score(mouse_result, keystroke_result, face_result)` — a weighted fusion function ready to accept teammates' keystroke/face module outputs once available. Currently defaults keystroke/face to neutral (0.5) when not provided.

**For integration, we need from keystroke/face teammates:**
- Their module's score in a similar `trust_score` (0.0-1.0) format
- Or tell us your output format and we'll adapt `fusion_engine.py` to match
## Project Structure

mouse-dynamics-auth/
- data/
  - raw/ (Balabit dataset + personal enrollment sessions)
  - processed/ (ML-ready feature CSVs)
- src/
  - data/ (preprocess.py, load_session.py)
  - features/ (extract_features.py, build_dataset.py)
  - models/ (train_baseline.py, train_personal.py, tune_model.py, compare_models.py)
  - api/ (app.py, predict.py, trust_engine.py, drift_handler.py; fusion_engine.py planned, not included)
- index.html (Live demo — mouse capture + model dropdown)
- enroll.html (Enrollment tool — records new genuine-user sessions)
- requirements.txt

## How to Run

source venv/bin/activate
python3 -m src.api.app

Then open index.html in a browser.

## Known Limitations (documented honestly for the report)
1. Personal model trained on limited enrollment data (~21K events / 6 sessions) — more sessions would improve robustness.
2. Behavior differs measurably between passive browsing and deliberate task-focused interaction (discovered during testing) — threshold calibrated to compensate; a larger, more varied enrollment dataset would be a more permanent fix.
3. Multimodal fusion is architecturally ready but not yet tested with real keystroke/face data.

## Debugging Log (useful for viva)
- Fixed severe class imbalance (23% -> 72%+ precision) via class_weight="balanced"
- Fixed window-size mismatch between training (50) and inference (100) that caused false impostor results
- Fixed overfitting on small personal dataset via max_depth/min_samples_leaf constraints
- Added CORS support for browser-to-API communication