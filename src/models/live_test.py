"""
AuthAI — Mouse Dynamics: LIVE TEST of the physical-mouse model   (NEW FILE)
==========================================================================
Place at:  src/models/live_test.py   (next to train_mouse_physical.py)
Run from project root:
    python src/models/live_test.py --genuine prerana
Then open  http://127.0.0.1:5050  in the same browser you recorded with.

What it does
  1. At start-up trains ONE model on ALL collected physical-mouse sessions
     (genuine user = 1, every other user = 0). Uses the same 12 features and
     200-event windows as the evaluation. Nothing on disk is changed.
  2. Serves a page that captures mouse events exactly like collect.html.
  3. Every 50 new events it scores the last 200 events (one window, ~3 s).
  4. Decision = mean of the last 20 window scores (~19 s of movement)
     compared with the threshold (0.42 = EER point found in evaluation).
Separate port (5050) so it never clashes with your existing app.py (5000).
"""
import argparse, os, sys, time
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mouse_physical import load_collected, extract_window_features  # EXISTING code reused

W, STEP = 200, 50
ap = argparse.ArgumentParser()
ap.add_argument("--genuine", required=True)
ap.add_argument("--data", default="data/raw/mouse")
ap.add_argument("--threshold", type=float, default=0.42)
ap.add_argument("--smooth", type=int, default=20)
args = ap.parse_args()

# ---------- 1. train on all sessions
sessions = load_collected(args.data)
X, y = [], []
for s in sessions:
    df = s["df"]
    for a in range(0, len(df) - W + 1, W // 2):
        X.append(extract_window_features(df.iloc[a:a + W]))
        y.append(1 if s["user"] == args.genuine else 0)
X, y = np.array(X), np.array(y)
clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                             class_weight="balanced", random_state=42, n_jobs=-1).fit(X, y)
print(f"Trained on {int(y.sum())} genuine + {int((y == 0).sum())} impostor windows "
      f"from {len({s['user'] for s in sessions})} people.")

# ---------- 2. API
app = Flask(__name__)
try:                                   # allow the team website (other port/origin) to call this API
    from flask_cors import CORS
    CORS(app)
except ImportError:
    print("flask-cors not installed: only same-origin calls will work (pip install flask-cors)")


@app.get("/health")
def health():
    return jsonify(status="ok", genuine_user=args.genuine, window_events=W,
                   threshold=args.threshold, smooth_windows=args.smooth)


@app.post("/score")
def score():
    ev = request.get_json()["events"]               # [[t, button, state, x, y], ...] last 200 events
    df = pd.DataFrame(ev, columns=["client timestamp", "button", "state", "x", "y"])
    p = float(clf.predict_proba(extract_window_features(df).reshape(1, -1))[0, 1])
    return jsonify(score=p, threshold=args.threshold)


# ---------- continuous authentication with the trust state kept on the server
STATE = defaultdict(lambda: deque(maxlen=args.smooth))   # session_id -> last `smooth` window scores


def _events_to_df(ev):
    """Accepts BOTH formats:
       list of lists  [[t, button, state, x, y], ...]                      (live_test page)
       list of dicts  [{"x":..,"y":..,"client timestamp":..,"state":..}, ...] (old index.html / app.py format)"""
    if isinstance(ev[0], dict):
        df = pd.DataFrame(ev)
        if "button" not in df.columns:
            df["button"] = "NoButton"
        return df[["client timestamp", "button", "state", "x", "y"]]
    return pd.DataFrame(ev, columns=["client timestamp", "button", "state", "x", "y"])


@app.post("/authenticate")
def authenticate():
    """Send the most recent mouse events (ideally the last 200) every ~50 new events.
    Body: {"events": [...], "session_id": "abc"}   (optional "reset": true to start fresh)"""
    data = request.get_json(silent=True) or {}
    ev, sid = data.get("events"), str(data.get("session_id", "default"))
    if data.get("reset"):
        STATE.pop(sid, None)
    if not ev or len(ev) < 50:
        return jsonify(error="send at least 50 events (recommended: the last 200)"), 400
    try:
        df = _events_to_df(ev[-W:])
        p = float(clf.predict_proba(extract_window_features(df).reshape(1, -1))[0, 1])
    except Exception as e:                              # malformed input -> clear error, no crash
        return jsonify(error=f"bad events: {e}"), 400
    q = STATE[sid]; q.append(p)
    trust = float(np.mean(q))
    if len(q) < args.smooth:
        decision, status = "collecting", "collecting"
    elif trust >= args.threshold:
        decision, status = "genuine", "trusted"
    else:
        decision, status = "impostor", "untrusted"
    return jsonify(
        modality="mouse", genuine_user=args.genuine, session_id=sid,
        mouse_confidence=round(p, 4), mouse_risk=round(1 - p, 4),          # this window only
        trust_score=round(trust, 4), trust_status=status, decision=decision, # use these for fusion
        threshold=args.threshold, windows_used=len(q), windows_required=args.smooth,
        timestamp=time.time())


@app.post("/reset")
def reset():
    STATE.pop(str((request.get_json(silent=True) or {}).get("session_id", "default")), None)
    return jsonify(status="reset")


@app.get("/")
def page():
    return PAGE.replace("__THR__", str(args.threshold)).replace("__K__", str(args.smooth)).replace("__USER__", args.genuine)


# ---------- 3. page (capture identical to collect.html: clientX/Y, mousemove/down/up/wheel)
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Mouse Live Test</title>
<style>body{margin:0;font:15px system-ui,sans-serif;background:#f6f7f9;color:#1d2330;user-select:none}
#panel{position:fixed;top:16px;left:50%;transform:translateX(-50%);background:#fff;border:1px solid #e3e6eb;border-radius:12px;padding:16px 24px;min-width:460px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06)}
#dec{font-size:30px;font-weight:800;margin:6px 0}.g{color:#14804a}.i{color:#c0392b}.w{color:#667085}
#bar{height:10px;background:#e3e6eb;border-radius:5px;position:relative;margin:10px 0}#fill{height:100%;border-radius:5px;background:#2f6fed;width:0}
#thr{position:absolute;top:-4px;width:2px;height:18px;background:#000}small{color:#667085}
button{margin-top:8px;padding:6px 14px;border-radius:6px;border:1px solid #ccd;background:#fff;cursor:pointer}
.t{position:absolute;width:40px;height:40px;border-radius:50%;background:#2f6fed;cursor:pointer}</style></head><body>
<div id="panel"><div>Enrolled user: <b>__USER__</b> · threshold __THR__</div>
<div id="dec" class="w">Move the mouse…</div>
<div id="bar"><div id="fill"></div><div id="thr"></div></div>
<div>Trust score (avg of last <span id="kk">0</span>/__K__ windows): <b id="sc">–</b> · last window: <span id="last">–</span></div>
<small>Events: <span id="n">0</span>. Use the mouse naturally, clicking the circles. A decision needs ~20 s of movement.</small><br>
<button onclick="reset()">Reset (new person)</button></div>
<script>
const THR=__THR__,K=__K__,W=200,STEP=50; let ev=[],sinceLast=0,scores=[],t0=performance.now(),busy=false;
document.getElementById('thr').style.left=(THR*100)+'%';
const now=()=>+((performance.now()-t0)/1000).toFixed(4), B={0:'Left',1:'Middle',2:'Right'};
function push(b,s,x,y){ev.push([now(),b,s,x,y]); if(ev.length>2000) ev=ev.slice(-1000); sinceLast++;
 document.getElementById('n').textContent=ev.length; if(ev.length>=W && sinceLast>=STEP && !busy){sinceLast=0; send();}}
addEventListener('mousemove',e=>push('NoButton',e.buttons?'Drag':'Move',e.clientX,e.clientY),true);
addEventListener('mousedown',e=>push(B[e.button]||'Other','Pressed',e.clientX,e.clientY),true);
addEventListener('mouseup',e=>push(B[e.button]||'Other','Released',e.clientX,e.clientY),true);
addEventListener('wheel',e=>push('Scroll',e.deltaY<0?'Up':'Down',e.clientX,e.clientY),{capture:true,passive:true});
async function send(){busy=true; try{const r=await fetch('/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({events:ev.slice(-W)})});
 const p=(await r.json()).score; scores.push(p); if(scores.length>K) scores.shift(); show(p);}catch(e){console.error(e)} busy=false;}
function show(p){const m=scores.reduce((a,b)=>a+b,0)/scores.length, d=document.getElementById('dec');
 document.getElementById('sc').textContent=m.toFixed(2); document.getElementById('last').textContent=p.toFixed(2);
 document.getElementById('kk').textContent=scores.length; document.getElementById('fill').style.width=(m*100)+'%';
 if(scores.length<K){d.textContent='Collecting… '+scores.length+'/'+K; d.className='w';}
 else if(m>=THR){d.textContent='GENUINE ✅'; d.className='g';} else {d.textContent='IMPOSTOR ❌'; d.className='i';}}
function reset(){ev=[];scores=[];sinceLast=0;t0=performance.now();document.getElementById('dec').textContent='Move the mouse…';document.getElementById('dec').className='w';
 document.getElementById('sc').textContent='–';document.getElementById('kk').textContent='0';document.getElementById('fill').style.width='0';}
function target(){const t=document.createElement('div');t.className='t';t.style.left=(40+Math.random()*(innerWidth-120))+'px';
 t.style.top=(180+Math.random()*(innerHeight-260))+'px';t.onclick=()=>{t.remove();target();};document.body.appendChild(t);} target();
</script></body></html>"""

if __name__ == "__main__":
    print("Open http://127.0.0.1:5050 in your browser.")
    app.run(port=5050, debug=False)
