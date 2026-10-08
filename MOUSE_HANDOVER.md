# AuthAI — Mouse Dynamics Module: Handover

**Module owner:** Prerana C M  ·  **Status:** working end to end (record → train → evaluate → live continuous authentication)

Read this file top to bottom once. Everything you need to run, test and integrate the mouse module is here.

---

## 1. What this module does

It checks whether the person using the mouse is the enrolled user (**genuine**) or someone else (**impostor**), continuously, from how they move the mouse.

```
mouse events (browser) → 200-event window (~3 s) → 12 movement features → Random Forest
→ score 0–1 per window → average of last 20 windows (~20 s) = trust score
→ trust score ≥ 0.42 → GENUINE, else IMPOSTOR
```

---

## 2. Folder contents (what each file is)

| File / folder | What it is | Touch it? |
|---|---|---|
| `collect.html` | Page used to record mouse data (genuine user and impostors) | No |
| `data/raw/mouse/` | Recorded physical-mouse sessions (`.csv` + `.json` per session), one folder per person: `prerana` = genuine (4 sessions), `imp01`–`imp04` = impostors (2 sessions each) | **Never edit** |
| `data/raw/my_data/`, `data/processed/` | Earlier trackpad enrollment data and feature CSVs (baseline experiment) | No |
| `models/`, `src/models/saved/` | Saved models. `live_test.py` does **not** load these; it retrains from `data/raw/mouse/` at start-up | No |
| `data/raw/Mouse-Dynamics-Challenge-master-2/` | Public Balabit dataset (used only for the comparison experiment) | No |
| `src/models/train_mouse_physical.py` | Loading, cleaning, the 12 features, training + FAR/FRR/EER | No |
| `src/models/evaluate_mouse_physical.py` | Final evaluation: unseen-impostor test, graphs for the paper | No |
| `src/models/live_test.py` | **The live module + API used for integration** (port 5050) | Only the settings in §6 |
| `results/mouse_physical_final/prerana/` | Final metrics + graphs | No |
| Older files (`src/api/app.py`, `index.html`, `enroll.html`, `src/api/*`, `src/features/*`, older `src/models/train_*.py`) | Previous trackpad version, kept as the baseline experiment. **Not used for integration** | **Do not delete** |

All files in the original trackpad pipeline are unchanged.

---

## 3. One-time setup

Requires Python 3.9+.

**Mac**
```
cd <path>/mouse-dynamics-auth
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install matplotlib flask-cors
```
**Windows**
```
cd <path>\mouse-dynamics-auth
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install matplotlib flask-cors
```
You should see `(venv)` at the start of the terminal line.

---

## 4. Run the live module

From the `mouse-dynamics-auth` folder, with `(venv)` active:
```
python src/models/live_test.py --genuine prerana
```
- It trains on the recorded data at start-up (~10 s), then prints `Running on http://127.0.0.1:5050`.
- Keep this terminal open.
- **Demo:** open `http://127.0.0.1:5050` in a browser, move the mouse and click the circles. After ~20 s it shows GENUINE ✅ or IMPOSTOR ❌. Click "Reset (new person)" before a different person tries.
- **Check it's alive:** open `http://127.0.0.1:5050/health` → `{"status":"ok", ...}`.

> ⚠️ **Must run on Prerana's MacBook with the Portronics mouse.** The model learned her movement on that laptop and mouse. On a different laptop or mouse it may reject her. To use another machine, Prerana must record 3–4 new sessions on it with `collect.html` (they go into `data/raw/mouse/prerana/`); restarting `live_test.py` then retrains automatically.

---

## 5. Integration API (for the website)

**Base URL:** `http://127.0.0.1:5050`  ·  CORS enabled (needs `flask-cors` installed)

> **Use `POST /authenticate` (below).** The server keeps the trust score for you, so the website only sends events and reads the decision.
> Do **not** use the old `src/api/app.py` (port 5000): that is the earlier trackpad version, kept only as a baseline.

### `GET /health`
```json
{"status":"ok","genuine_user":"prerana","window_events":200,"threshold":0.42,"smooth_windows":20}
```

### `POST /authenticate`  ← main integration endpoint
Call it **every 50 new mouse events**, sending the **last 200 events**.

Request (either event format works):
```json
{ "session_id": "user-session-123",
  "events": [ [t, "button", "state", x, y], ... ] }
```
or the old format used by `index.html`:
```json
{ "session_id": "user-session-123",
  "events": [ {"x":100, "y":200, "client timestamp":0.50, "state":"Move"}, ... ] }
```
Add `"reset": true` once when a new person or login session starts (or call `POST /reset` with `{"session_id": ...}`).

Response:
```json
{
  "modality": "mouse",
  "genuine_user": "prerana",
  "session_id": "user-session-123",
  "mouse_confidence": 0.80,
  "mouse_risk": 0.20,
  "trust_score": 0.70,
  "trust_status": "trusted",
  "decision": "genuine",
  "threshold": 0.42,
  "windows_used": 20,
  "windows_required": 20,
  "timestamp": 1791415788.11
}
```
| Field | Meaning |
|---|---|
| `mouse_confidence` | score of this single window (~3 s); noisy, so don't decide on it |
| `trust_score` | **average of the last 20 windows (~20 s): use this** |
| `decision` | `"collecting"` (fewer than 20 windows yet), `"genuine"` or `"impostor"` |
| `trust_status` | `"collecting"`, `"trusted"` or `"untrusted"` |
| `windows_used` | how many windows are in the average so far (0–20) |

Errors return HTTP 400 with `{"error": "..."}` (for example, fewer than 50 events).

### `POST /score` (low-level, optional)
Scores **one window = the last 200 mouse events**.

Request:
```json
{ "events": [ [t, "button", "state", x, y], ... 200 items ... ] }
```
| Field | Meaning |
|---|---|
| `t` | seconds since capture started (`performance.now()/1000`), increasing |
| `button` | `"NoButton"`, `"Left"`, `"Right"`, or `"Scroll"` |
| `state` | `"Move"`, `"Drag"`, `"Pressed"`, `"Released"`, `"Up"`, `"Down"` |
| `x`, `y` | `event.clientX`, `event.clientY` (CSS pixels) |

Response:
```json
{ "score": 0.79, "threshold": 0.42 }
```
`score` = probability that this window came from the enrolled user (0–1).

### Capturing events on the website (exactly what the demo page does)
The steps below describe capture plus client-side averaging with `/score`. With `/authenticate`, do steps 1–2 only (POST to `/authenticate` instead of `/score`); the server does steps 3–5.

1. Listen on the page for `mousemove`, `mousedown`, `mouseup`, `wheel` and store each event as above:
   - `mousemove` → `["NoButton", e.buttons ? "Drag" : "Move", x, y]`
   - `mousedown` / `mouseup` → `["Left"/"Right", "Pressed"/"Released", x, y]`
   - `wheel` → `["Scroll", deltaY < 0 ? "Up" : "Down", x, y]`
2. Once there are ≥ 200 events, **every 50 new events** POST the **last 200** to `/score`.
3. Keep the **last 20 scores**; `trust_score = average(last 20)`.
4. Fewer than 20 scores → status **"collecting"** (no decision yet).
5. `trust_score ≥ 0.42` → **genuine**, else **impostor**.

The JavaScript to copy is inside `live_test.py` (the `PAGE` string, `<script>` section): it is about 20 lines and already does steps 1–5.

---

## 6. Settings (command-line flags of `live_test.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--genuine` | (required) | enrolled user ID, `prerana` |
| `--threshold` | `0.42` | decision threshold (EER point from evaluation). Higher = stricter (fewer impostors accepted, more false rejections of Prerana) |
| `--smooth` | `20` | number of windows averaged (~20 s). More = more stable but slower |

---

## 7. Output for multimodal fusion (face + keystroke + mouse)

The `/authenticate` response is already the fusion output. Use these fields:
```json
{ "modality": "mouse", "trust_score": 0.70, "threshold": 0.42,
  "decision": "genuine" | "impostor" | "collecting", "windows_used": 0-20, "timestamp": ... }
```
- For score-level fusion, use `trust_score`. It is already 0–1, same direction as a "genuine" probability.
- When `decision` is `"collecting"`, the mouse score is not ready yet: leave it out of the fusion or give it zero weight.
- The mouse module is the weakest when there is little movement. It works best as a continuous background signal, not a one-shot login check.

---

## 8. Results (for report and paper)

Data: 1 genuine user (4 sessions), 4 impostors (2 sessions each); same MacBook, same Portronics mouse, same browser page. Split by session (no leakage).

| Test | Decision time | EER | AUC |
|---|---|---|---|
| Unseen impostor (never in training) | ~3 s | 0.330 ± 0.100 | 0.743 |
| **Unseen impostor** | **~19 s** | **0.298 ± 0.162** | **0.791** |
| Known impostor, new session | ~19 s | 0.308 ± 0.047 | 0.816 |

Per impostor (unseen, ~19 s): imp03 EER 0.146 · imp04 0.192 · imp02 0.318 · imp01 0.537 (moves similarly to the genuine user).

**Key finding:** a model trained with public Balabit data as impostors scored a perfect EER of 0.000 on Balabit, but accepted a real unseen person on the same laptop **98.8%** of the time (AUC 0.49). It had learned the recording setup, not the person. Impostor data must therefore come from the same device and setup.

Graphs: `results/mouse_physical_final/prerana/final_evaluation.png`, `per_impostor_eer.png`.
Reproduce: `python src/models/evaluate_mouse_physical.py --genuine prerana`

---

## 9. Known limitations

- Small dataset: 1 genuine user, 4 impostors, all recorded on one day.
- About 70% correct decisions at ~20 s (EER ≈ 0.30). Usable as one signal in fusion, not as the only authentication.
- Device-specific: retrain if the laptop or mouse changes (see §4).

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named ...` | `(venv)` not active, so activate it (§3) |
| `No module named flask_cors` | `pip install flask-cors` |
| Port 5050 already in use | close the other terminal running `live_test.py` |
| Browser shows CORS error | install `flask-cors`, restart `live_test.py` |
| Prerana always rejected | wrong laptop/mouse, or mouse DPI changed; see the warning in §4 |
| `--genuine prerana` finds no data | run the command from inside the `mouse-dynamics-auth` folder |

Questions: Prerana C M (Mouse Dynamics module).
