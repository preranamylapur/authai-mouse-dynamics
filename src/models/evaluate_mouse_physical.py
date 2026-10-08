"""
AuthAI — Mouse Dynamics: FINAL biometric evaluation (physical mouse)   (NEW FILE)
================================================================================
Place at:  src/models/evaluate_mouse_physical.py   (next to train_mouse_physical.py)
Run from project root:
    python src/models/evaluate_mouse_physical.py --genuine prerana

Reuses the loading + 12-feature extraction from train_mouse_physical.py (no duplicate code).
Does NOT modify any data or model. Writes only results/mouse_physical_final/<user>/.

Two evaluation protocols (both split by SESSION -> no leakage):
  P1  Unseen impostor (leave-one-impostor-out):
      for each impostor u: train on genuine sessions 1-2 + ALL OTHER impostors;
      test genuine held-out session vs u, who was never seen in training.
      -> answers "is a stranger rejected without ever training on them?"
  P2  Known impostor, new session:
      train on genuine 1-2 + every impostor's session 1 (or 2); test on the other session.
Folds rotate over genuine test session (3, 4) -> results reported as mean ± std.

Continuous authentication: test windows slide every 50 events; the decision score is the
mean of the last k window scores (k=1 -> one 200-event window ≈ 3 s; k=20 ≈ 19 s of movement).
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mouse_physical import load_collected, extract_window_features, FEATURES  # EXISTING code reused

SEED, W, SLIDE, KS = 42, 200, 50, [1, 5, 10, 20, 40]


def windows(df, step):
    return np.array([extract_window_features(df.iloc[a:a + W]) for a in range(0, len(df) - W + 1, step)])


def eer_auc(gen, imp):
    """FAR(t)=P(impostor score>=t), FRR(t)=P(genuine score<t); EER where FAR=FRR."""
    y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]; s = np.r_[gen, imp]
    far, tpr, thr = roc_curve(y, s); frr = 1 - tpr; i = np.argmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2, roc_auc_score(y, s), far, frr, thr


def smooth(s, k):
    return np.convolve(s, np.ones(k) / k, "valid") if k > 1 and len(s) >= k else s


def model():
    return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genuine", required=True)
    ap.add_argument("--data", default="data/raw/mouse")
    ap.add_argument("--results", default="results/mouse_physical_final")
    a = ap.parse_args()
    out = os.path.join(a.results, a.genuine); os.makedirs(out, exist_ok=True)

    S = load_collected(a.data)
    D = {(s["user"], s["session"]): s["df"] for s in S}
    gsess = sorted(s for (u, s) in D if u == a.genuine)
    imps = sorted({u for (u, _) in D if u != a.genuine})
    assert len(gsess) >= 3 and len(imps) >= 2, "need >=3 genuine sessions and >=2 impostors"
    gtest = gsess[2:]                     # e.g. sessions 3,4 (and any later ones) are test folds
    gtrain = gsess[:2]
    print(f"genuine '{a.genuine}' sessions {gsess} (train {gtrain}, test folds {gtest}); impostors {imps}")
    cache = {}
    def X(key, step):
        if (key, step) not in cache: cache[(key, step)] = windows(D[key], step)
        return cache[(key, step)]
    Xg_train = np.vstack([X((a.genuine, s), W // 2) for s in gtrain])

    # ---------------- P1: unseen impostor (leave-one-impostor-out)
    p1 = {k: [] for k in KS}; per_imp = {u: [] for u in imps}; pooled = {"g": [], "i": []}
    for gt in gtest:
        for u in imps:
            Xi = np.vstack([X(k, W // 2) for k in D if k[0] != a.genuine and k[0] != u])
            clf = model().fit(np.vstack([Xg_train, Xi]), np.r_[np.ones(len(Xg_train)), np.zeros(len(Xi))])
            sg = clf.predict_proba(X((a.genuine, gt), SLIDE))[:, 1]
            si = [clf.predict_proba(X(k, SLIDE))[:, 1] for k in D if k[0] == u]
            for k in KS:
                e, au, *_ = eer_auc(smooth(sg, k), np.concatenate([smooth(x, k) for x in si]))
                p1[k].append((e, au))
            per_imp[u].append(eer_auc(smooth(sg, 20), np.concatenate([smooth(x, 20) for x in si]))[0])
            pooled["g"].append(smooth(sg, 20)); pooled["i"] += [smooth(x, 20) for x in si]

    # ---------------- P2: known impostor, new session
    p2 = {k: [] for k in KS}
    isess = sorted({s for (u, s) in D if u != a.genuine})
    for gt in gtest:
        for ts in isess[:2]:
            tr = [k for k in D if k[0] != a.genuine and k[1] != ts]
            Xi = np.vstack([X(k, W // 2) for k in tr])
            clf = model().fit(np.vstack([Xg_train, Xi]), np.r_[np.ones(len(Xg_train)), np.zeros(len(Xi))])
            sg = clf.predict_proba(X((a.genuine, gt), SLIDE))[:, 1]
            si = [clf.predict_proba(X((u, ts), SLIDE))[:, 1] for u in imps if (u, ts) in D]
            for k in KS:
                e, au, *_ = eer_auc(smooth(sg, k), np.concatenate([smooth(x, k) for x in si]))
                p2[k].append((e, au))

    # ---------------- table
    secs = lambda k: (W + SLIDE * (k - 1)) / 60.0          # ~60 events per second
    rows, res = [], {"genuine": a.genuine, "impostors": imps, "window_events": W, "protocols": {}}
    for name, P in [("P1 unseen impostor", p1), ("P2 known impostor, new session", p2)]:
        res["protocols"][name] = {}
        for k in KS:
            r = np.array(P[k])
            res["protocols"][name][f"k={k}"] = dict(seconds=round(secs(k), 1), eer_mean=float(r[:, 0].mean()),
                                                   eer_std=float(r[:, 0].std()), auc_mean=float(r[:, 1].mean()))
            rows.append(f"| {name} | {secs(k):.0f} s | {r[:,0].mean():.3f} ± {r[:,0].std():.3f} | {r[:,1].mean():.3f} |")
    res["per_impostor_eer_20win"] = {u: float(np.mean(v)) for u, v in per_imp.items()}
    table = "| Protocol | Decision time | EER (mean ± std) | AUC |\n|---|---|---|---|\n" + "\n".join(rows)
    print("\n" + table + "\n\nPer-impostor EER (P1, ~19 s):", {u: round(v, 3) for u, v in res["per_impostor_eer_20win"].items()})
    json.dump(res, open(os.path.join(out, "final_metrics.json"), "w"), indent=2)
    open(os.path.join(out, "final_metrics.md"), "w").write(table + "\n")

    # ---------------- figures
    g = np.concatenate(pooled["g"]); i = np.concatenate(pooled["i"])
    e, au, far, frr, thr = eer_auc(g, i)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(far, 1 - frr, lw=2); ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].scatter([e], [1 - e], c="red", zorder=5, label=f"EER ≈ {e:.3f}")
    ax[0].set(xlabel="FAR", ylabel="1 − FRR", title=f"ROC, all folds pooled, unseen impostors, ~19 s (AUC {au:.3f})"); ax[0].legend(loc="lower right")
    b = np.linspace(0, 1, 26)
    ax[1].hist(g, b, alpha=.6, density=True, label="genuine"); ax[1].hist(i, b, alpha=.6, density=True, label="impostor (unseen)")
    ax[1].set(xlabel="Authentication score P(genuine)", ylabel="Density", title="Score distributions"); ax[1].legend()
    for name, P, m in [("unseen impostor (P1)", p1, "o-"), ("known impostor, new session (P2)", p2, "s--")]:
        mu = [np.mean([x[0] for x in P[k]]) for k in KS]; sd = [np.std([x[0] for x in P[k]]) for k in KS]
        ax[2].errorbar([secs(k) for k in KS], mu, yerr=sd, fmt=m, capsize=3, label=name)
    ax[2].set(xlabel="Seconds of mouse movement per decision", ylabel="EER", title="EER vs decision length", ylim=(0, 0.6)); ax[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "final_evaluation.png"), dpi=150); plt.close(fig)

    plt.figure(figsize=(6, 3.8))
    us = list(res["per_impostor_eer_20win"]); plt.bar(us, [res["per_impostor_eer_20win"][u] for u in us])
    plt.axhline(0.5, ls=":", c="grey"); plt.ylabel("EER (lower = easier to reject)")
    plt.title("Per-impostor EER, never seen in training (~19 s)"); plt.tight_layout()
    plt.savefig(os.path.join(out, "per_impostor_eer.png"), dpi=150); plt.close()
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
