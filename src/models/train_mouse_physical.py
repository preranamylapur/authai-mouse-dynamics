"""
AuthAI — Mouse Dynamics: Physical-mouse training + biometric evaluation  (NEW FILE)
================================================================================
Place at:  src/models/train_mouse_physical.py
Run from project root:
    python -m src.models.train_mouse_physical --genuine prerana
    # if no human impostors could be collected, add Balabit as impostors:
    python -m src.models.train_mouse_physical --genuine prerana --balabit data/raw/balabit/training_files

Does NOT modify any existing file, model or dataset. Writes only:
    models/mouse_physical_<user>.pkl
    results/mouse_physical/<user>/  (metrics.json, metrics.md, *.png)

Pipeline
    raw CSV (Balabit 6-column format, from collect.html)
      -> clean (sort by time, drop duplicates / zero-time-delta events)
      -> 100-event windows
      -> 12 behavioural features per window (same set as the existing module)
      -> SESSION-level split (no windows from one session in both train and test)
      -> Random Forest (class_weight=balanced, depth-limited against overfitting)
      -> genuine/impostor scores -> FAR, FRR, EER, ROC-AUC, plots
"""
import argparse, glob, json, logging, os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

SEED = 42
WINDOW = 100          # events per window — must equal the live demo's window size
TRAIN_STEP = 50       # 50% overlap for training windows (more samples)
TEST_STEP = 100       # no overlap for test windows (independent samples)
SMOOTH_K = 5          # consecutive windows averaged = your trust engine's 5-window history
FEATURES = ["total_distance", "avg_velocity", "max_velocity", "min_velocity",
            "avg_acceleration", "avg_jerk", "avg_direction_change", "straightness",
            "click_count", "click_frequency", "velocity_variance", "duration"]
log = logging.getLogger("train_mouse_physical")


# ---------------------------------------------------------------- 1. LOADING
def load_session(csv_path):
    """Read one session CSV and clean it."""
    df = pd.read_csv(csv_path).dropna()
    df = df.sort_values("client timestamp", kind="stable").reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_collected(root):
    """data/raw/mouse/<user>/<file>.csv (+ .json sidecar from collect.html)."""
    sessions = []
    for csv in sorted(glob.glob(os.path.join(root, "*", "*.csv"))):
        meta_path = csv[:-4] + ".json"
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        user = meta.get("user_id", os.path.basename(os.path.dirname(csv)))
        sess = meta.get("session", len([s for s in sessions if s["user"] == user]) + 1)
        sessions.append(dict(user=user, session=int(sess), source="collected",
                             device=meta.get("device", "mouse"), df=load_session(csv)))
    return sessions


def load_balabit(root, max_sessions_per_user=3):
    """Balabit training_files/<user>/session_* — used only as fallback impostors."""
    out = []
    for udir in sorted(glob.glob(os.path.join(root, "user*"))):
        for i, f in enumerate(sorted(glob.glob(os.path.join(udir, "session_*")))[:max_sessions_per_user]):
            out.append(dict(user="balabit_" + os.path.basename(udir), session=i + 1,
                            source="balabit", device="unknown", df=load_session(f)))
    return out


# ------------------------------------------------------- 2. FEATURE EXTRACTION
def extract_window_features(w):
    """12 features for one window of raw events (DataFrame with x, y, client timestamp, state).
    Kinematics use only Move/Drag rows; zero time-delta steps are removed
    (the duplicate-timestamp velocity-spike fix from the progress report)."""
    t_all = w["client timestamp"].to_numpy(float)
    duration = float(t_all[-1] - t_all[0])
    clicks = int((w["state"] == "Pressed").sum())

    m = w[w["state"].isin(["Move", "Drag"])]
    x, y, t = m["x"].to_numpy(float), m["y"].to_numpy(float), m["client timestamp"].to_numpy(float)
    dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
    keep = dt > 0
    dx, dy, dt = dx[keep], dy[keep], dt[keep]
    dist = np.hypot(dx, dy)
    total = float(dist.sum())

    if len(dt) >= 3:
        v = dist / dt                                   # speed  (px/s)
        tm = t[1:][keep]                                # timestamps aligned with v
        dtv = np.diff(tm); dtv[dtv <= 0] = np.nan
        a = np.diff(v) / dtv                            # acceleration (px/s^2)
        j = np.diff(a) / dtv[1:]                        # jerk (px/s^3)
        theta = np.arctan2(dy, dx)
        dtheta = np.abs(np.angle(np.exp(1j * np.diff(theta))))   # wrapped to [0, pi]
        moving = dist > 0
        dtheta = dtheta[moving[1:] & moving[:-1]]
        net = np.hypot(x[-1] - x[0], y[-1] - y[0])
        feats = [total, np.mean(v), np.max(v), np.min(v), np.nanmean(a), np.nanmean(j),
                 np.mean(dtheta) if len(dtheta) else 0.0, net / total if total > 0 else 0.0,
                 clicks, clicks / duration if duration > 0 else 0.0, np.var(v), duration]
    else:
        feats = [total, 0, 0, 0, 0, 0, 0, 0, clicks, clicks / duration if duration > 0 else 0, 0, duration]
    return np.nan_to_num(np.array(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def windows(df, step):
    rows = [extract_window_features(df.iloc[s:s + WINDOW]) for s in range(0, len(df) - WINDOW + 1, step)]
    return np.array(rows) if rows else np.empty((0, len(FEATURES)))


# ------------------------------------------------------------- 3. SPLITTING
def split(sessions, genuine, force_unseen=()):
    """Session-level split. Returns dict of lists of sessions.
    genuine:  oldest sessions -> train, next -> validation (threshold), newest -> test
    impostors: users split in two groups -> 'seen' (in training) and 'unseen' (test only)
    """
    rng = np.random.RandomState(SEED)
    g = sorted([s for s in sessions if s["user"] == genuine], key=lambda s: s["session"])
    if len(g) < 3:
        raise SystemExit(f"Need at least 3 genuine sessions for '{genuine}', found {len(g)}.")
    n_test = 1
    n_val = 1
    parts = dict(g_train=g[:-(n_test + n_val)], g_val=g[-(n_test + n_val):-n_test], g_test=g[-n_test:])

    imp_users = sorted({s["user"] for s in sessions if s["user"] != genuine})
    if not imp_users:
        raise SystemExit("No impostor data. Collect impostors or pass --balabit <path>.")
    rng.shuffle(imp_users)
    if force_unseen:                                  # --unseen: these people are NEVER used in training
        unseen = [u for u in imp_users if u in force_unseen]
        seen = [u for u in imp_users if u not in force_unseen]
        if not seen:
            raise SystemExit("All impostors are --unseen; add --balabit (or more people) as training background.")
    elif len(imp_users) >= 2:
        n_seen = max(1, len(imp_users) // 2)
        seen, unseen = imp_users[:n_seen], imp_users[n_seen:]
    else:
        seen, unseen = imp_users, []
    parts.update(i_train=[], i_val=[], i_test_seen=[], i_test_unseen=[])
    for u in seen:
        us = sorted([s for s in sessions if s["user"] == u], key=lambda s: s["session"])
        if len(us) == 1:
            parts["i_train"] += us                        # can't hold out a session
        elif len(us) == 2:
            parts["i_train"].append(us[0]); parts["i_test_seen"].append(us[1])
        else:
            parts["i_train"] += us[:-2]; parts["i_val"].append(us[-2]); parts["i_test_seen"].append(us[-1])
    for u in unseen:
        parts["i_test_unseen"] += [s for s in sessions if s["user"] == u]
    if not parts["i_val"]:                                # borrow an unseen impostor session for threshold selection
        if parts["i_test_unseen"]:
            parts["i_val"].append(parts["i_test_unseen"][0])
    parts["seen_users"], parts["unseen_users"] = seen, unseen
    return parts


def stack(sess_list, step):
    X = [windows(s["df"], step) for s in sess_list]
    sizes = [len(x) for x in X]
    return (np.vstack(X) if X else np.empty((0, len(FEATURES)))), sizes


# --------------------------------------------------------------- 4. METRICS
def biometric_metrics(gen, imp):
    """gen/imp = scores P(genuine) for genuine and impostor windows.
    FAR(t) = fraction of impostor scores >= t   (impostor wrongly accepted)
    FRR(t) = fraction of genuine  scores <  t   (genuine wrongly rejected)
    EER    = error rate where FAR == FRR."""
    y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    s = np.r_[gen, imp]
    far, tpr, thr = roc_curve(y, s)
    frr = 1 - tpr
    i = np.nanargmin(np.abs(far - frr))
    return dict(eer=float((far[i] + frr[i]) / 2), eer_threshold=float(min(thr[i], 1.0)),
                auc=float(roc_auc_score(y, s)), n_genuine=int(len(gen)), n_impostor=int(len(imp))), (far, frr, thr)


def at_threshold(gen, imp, t):
    far = float(np.mean(imp >= t)) if len(imp) else float("nan")
    frr = float(np.mean(gen < t)) if len(gen) else float("nan")
    acc = float((np.sum(gen >= t) + np.sum(imp < t)) / (len(gen) + len(imp)))
    return dict(threshold=float(t), FAR=far, FRR=frr, accuracy=acc, balanced_accuracy=1 - (far + frr) / 2)


def smooth(scores, sizes, k=SMOOTH_K):
    """Average k consecutive windows within each session (continuous authentication)."""
    out, i = [], 0
    for n in sizes:
        s = scores[i:i + n]; i += n
        if n >= k:
            out.extend(np.convolve(s, np.ones(k) / k, mode="valid"))
    return np.array(out)


# ----------------------------------------------------------------- 5. PLOTS
def _plt():
    """matplotlib is imported only when plotting, so the live API does not need it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plots(outdir, gen, imp, curves, eer, thr_op, title):
    plt = _plt()
    far, frr, thr = curves
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(far, 1 - frr, lw=2); ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].scatter([eer["eer"]], [1 - eer["eer"]], c="red", zorder=5, label=f"EER = {eer['eer']:.3f}")
    ax[0].set(xlabel="False Acceptance Rate (FAR)", ylabel="1 − FRR (True Accept Rate)",
              title=f"ROC  (AUC = {eer['auc']:.3f})"); ax[0].legend(loc="lower right")
    t = np.clip(thr, 0, 1)
    ax[1].plot(t, far, label="FAR"); ax[1].plot(t, frr, label="FRR")
    ax[1].axvline(eer["eer_threshold"], ls=":", c="red", label="EER point")
    ax[1].axvline(thr_op, ls="--", c="black", label=f"operating threshold {thr_op:.2f}")
    ax[1].set(xlabel="Threshold on P(genuine)", ylabel="Error rate", title="FAR / FRR vs threshold", xlim=(0, 1)); ax[1].legend()
    bins = np.linspace(0, 1, 26)
    ax[2].hist(gen, bins, alpha=.6, label="genuine", density=True)
    ax[2].hist(imp, bins, alpha=.6, label="impostor", density=True)
    ax[2].axvline(thr_op, ls="--", c="black")
    ax[2].set(xlabel="Score P(genuine)", ylabel="Density", title="Score distributions"); ax[2].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(outdir, "evaluation.png"), dpi=150); plt.close(fig)


# ------------------------------------------------------------------- MAIN
def main():
    global WINDOW, TRAIN_STEP, TEST_STEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--genuine", required=True, help="user_id of the account owner, e.g. prerana")
    ap.add_argument("--data", default="data/raw/mouse")
    ap.add_argument("--balabit", default=None, help="Balabit training_files path (fallback impostors)")
    ap.add_argument("--unseen", nargs="*", default=[], help="impostor user_ids kept out of training (test only)")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results/mouse_physical")
    ap.add_argument("--window", type=int, default=WINDOW, help="events per window (live demo must use the same)")
    args = ap.parse_args()
    WINDOW, TRAIN_STEP, TEST_STEP = args.window, args.window // 2, args.window
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    np.random.seed(SEED)

    sessions = load_collected(args.data)
    if args.balabit:
        sessions += load_balabit(args.balabit)
    for s in sessions:
        log.info(f"loaded {s['user']:<22} session {s['session']:<3} {s['source']:<9} {len(s['df']):>6} events")

    p = split(sessions, args.genuine, tuple(args.unseen))
    if not (p["i_test_seen"] or p["i_test_unseen"]):
        raise SystemExit("No impostor TEST session: each impostor needs 2 sessions, or use --unseen <id> with --balabit.")
    log.info(f"genuine train/val/test sessions: {[s['session'] for s in p['g_train']]} / "
             f"{[s['session'] for s in p['g_val']]} / {[s['session'] for s in p['g_test']]}")
    log.info(f"impostors seen in training: {p['seen_users']}   unseen (test only): {p['unseen_users']}")

    Xg, _ = stack(p["g_train"], TRAIN_STEP); Xi, _ = stack(p["i_train"], TRAIN_STEP)
    X = np.vstack([Xg, Xi]); y = np.r_[np.ones(len(Xg)), np.zeros(len(Xi))]
    log.info(f"training windows: genuine {len(Xg)}, impostor {len(Xi)}")

    clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                 class_weight="balanced", random_state=SEED, n_jobs=-1)
    clf.fit(X, y)
    score = lambda lst: (clf.predict_proba(stack(lst, TEST_STEP)[0])[:, 1] if lst and len(stack(lst, TEST_STEP)[0]) else np.array([]))

    # Threshold chosen on VALIDATION data only (never on test) -> EER point of validation
    gv, iv = score(p["g_val"]), score(p["i_val"])
    thr = biometric_metrics(gv, iv)[0]["eer_threshold"] if len(gv) and len(iv) else 0.5
    log.info(f"operating threshold (from validation EER): {thr:.3f}")

    gt = score(p["g_test"]); its = score(p["i_test_seen"]); itu = score(p["i_test_unseen"])
    results = {"genuine_user": args.genuine, "window_events": WINDOW, "operating_threshold": thr,
               "seen_impostors": p["seen_users"], "unseen_impostors": p["unseen_users"],
               "balabit_used": bool(args.balabit), "evaluations": {}}
    outdir = os.path.join(args.results, args.genuine); os.makedirs(outdir, exist_ok=True)

    evals = {"all_impostors": np.r_[its, itu], "seen_impostors": its, "unseen_impostors": itu}
    for name, imp in evals.items():
        if not len(imp) or not len(gt):
            continue
        m, curves = biometric_metrics(gt, imp)
        m.update(at_threshold(gt, imp, thr))
        # continuous authentication: average of 5 consecutive windows (like the trust engine)
        g_sizes = stack(p["g_test"], TEST_STEP)[1]
        imp_lists = {"all_impostors": p["i_test_seen"] + p["i_test_unseen"],
                     "seen_impostors": p["i_test_seen"], "unseen_impostors": p["i_test_unseen"]}[name]
        gs, is_ = smooth(gt, g_sizes), smooth(imp, stack(imp_lists, TEST_STEP)[1])
        if len(gs) and len(is_):
            ms, _ = biometric_metrics(gs, is_)
            m["smoothed_5win_eer"], m["smoothed_5win_auc"] = ms["eer"], ms["auc"]
        results["evaluations"][name] = m
        if name == "all_impostors":
            plots(outdir, gt, imp, curves, m, thr, f"Physical mouse — genuine '{args.genuine}' vs impostors (test sessions)")

    # feature importance plot
    plt = _plt()
    imp_order = np.argsort(clf.feature_importances_)
    plt.figure(figsize=(7, 4.5)); plt.barh(np.array(FEATURES)[imp_order], clf.feature_importances_[imp_order])
    plt.title("Random Forest feature importance"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_importance.png"), dpi=150); plt.close()

    # save model + everything needed to use it live
    os.makedirs(args.models, exist_ok=True)
    model_path = os.path.join(args.models, f"mouse_physical_{args.genuine}.pkl")
    joblib.dump({"model": clf, "features": FEATURES, "window": WINDOW, "threshold": thr,
                 "device": "mouse", "genuine_user": args.genuine}, model_path)
    json.dump(results, open(os.path.join(outdir, "metrics.json"), "w"), indent=2)

    lines = ["| Test set | Genuine win. | Impostor win. | EER | AUC | FAR @thr | FRR @thr | Accuracy | EER (5-window smoothed) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, m in results["evaluations"].items():
        lines.append(f"| {k} | {m['n_genuine']} | {m['n_impostor']} | {m['eer']:.3f} | {m['auc']:.3f} | "
                     f"{m['FAR']:.3f} | {m['FRR']:.3f} | {m['accuracy']:.3f} | {m.get('smoothed_5win_eer', float('nan')):.3f} |")
    table = "\n".join(lines)
    open(os.path.join(outdir, "metrics.md"), "w").write(f"Operating threshold: {thr:.3f}\n\n{table}\n")
    print("\n" + table + f"\n\nOperating threshold: {thr:.3f}\nModel: {model_path}\nResults: {outdir}/")


if __name__ == "__main__":
    main()
