"""
Benchmark this project against EVDetect (Lazaridis et al., Scientific Reports 2026,
https://github.com/kosmas15/EVDetect) on the same CICEVSE2024 EVSE-B power file.

Two protocols are run for every scenario:

  A  "evdetect"  - EVDetect's exact protocol: stratified random row split
                   (80/20 test, then 80/20 train/val, random_state=42), raw
                   4 physical features, row-level bootstrap CIs (1,000 resamples).
                   This reproduces their test set row-for-row.
  B  "block"     - leakage-aware protocol from this project: whole 1-minute
                   blocks go to train or test (80/20, random_state=42), and
                   CIs come from a block bootstrap (resampling test minutes).

Scenarios (as in EVDetect): binary / 5-class multiclass, each on the full
dataset and on the charging-state subset. Multiclass uses EVDetect's labels:
Benign, DoS, Recon, Backdoor, Cryptojacking.

Models:
  EVDetect RF / EVDetect XGBoost - their tuned best hyperparameters per scenario
  Ours (full pipeline)           - rolling features + Isolation Forest score +
                                   XGB/LGBM/RF soft vote (binary) or XGBoost (multiclass)
  Ours (raw features)            - same models on the raw 4 signals only (ablation)

Run from the repository root:
    python benchmarks/evdetect/run_benchmark.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.pipeline import DATA_PATH, METADATA, SIGNALS, engineer  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "results"
SEED = 42
N_BOOT = 1000
RAW = SIGNALS  # shunt_voltage, bus_voltage_V, current_mA, power_mW

ATTACK_MAP = {  # EVDetect's 5-class mapping
    "syn-flood": "DoS", "tcp-flood": "DoS", "cryptojacking": "Cryptojacking",
    "vuln-scan": "Recon", "syn-stealth": "Recon", "Backdoor": "Backdoor", "none": "Benign",
}

# Best hyperparameters reported in EVDetect's notebooks (03_binary, 02_multiclass)
EVD_RF = {
    ("binary", "full"): dict(class_weight={0: 2, 1: 1}, max_depth=20, max_features="sqrt",
                             min_samples_leaf=2, min_samples_split=2, n_estimators=200),
    ("binary", "charging"): dict(class_weight="balanced", max_depth=None, max_features="sqrt",
                                 min_samples_leaf=2, min_samples_split=5, n_estimators=100),
    ("multiclass", "full"): dict(class_weight=None, max_depth=20, max_features="sqrt",
                                 min_samples_leaf=2, min_samples_split=2, n_estimators=200),
    ("multiclass", "charging"): dict(class_weight="balanced", max_depth=20, max_features="sqrt",
                                     min_samples_leaf=2, min_samples_split=5, n_estimators=200),
}
EVD_XGB = {
    ("binary", "full"): dict(colsample_bytree=0.8, gamma=0.5, learning_rate=0.05, max_depth=8,
                             min_child_weight=3, n_estimators=400, scale_pos_weight=0.7, subsample=0.8),
    ("binary", "charging"): dict(colsample_bytree=0.8, gamma=0.5, learning_rate=0.1, max_depth=5,
                                 n_estimators=200, subsample=0.8),
    ("multiclass", "full"): dict(colsample_bytree=1.0, gamma=0.5, learning_rate=0.1, max_depth=7,
                                 n_estimators=200, subsample=0.8),
    ("multiclass", "charging"): dict(colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_depth=5,
                                     n_estimators=200, subsample=0.8),
}
# EVDetect's published bootstrap point estimates (05_bootstrap_cis.ipynb) for reference
EVD_PUBLISHED = {
    ("binary", "full"): {"Random Forest": (0.9454, 0.8727), "XGBoost": (0.9475, 0.8774), "Stacking": (0.9481, 0.8739)},
    ("binary", "charging"): {"Random Forest": (0.8618, 0.8096), "XGBoost": (0.8731, 0.8103), "Stacking": (0.8586, 0.8130)},
    ("multiclass", "full"): {"Random Forest": (0.6311, 0.6516), "XGBoost": (0.6441, 0.6551), "Stacking": (0.6432, 0.6587)},
    ("multiclass", "charging"): {"Random Forest": (0.8493, 0.8397), "XGBoost": (0.8571, 0.8422), "Stacking": (0.8539, 0.8439)},
}


# --------------------------------------------------------------------------- #
def metrics_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    pred = cm.sum(0).astype(float)
    true = cm.sum(1).astype(float)
    present = true > 0
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        "accuracy": tp.sum() / cm.sum(),
        "precision": prec[present].mean(),
        "recall": rec[present].mean(),       # = balanced accuracy
        "f1": f1[present].mean(),
    }


def cm_of(y, p, k):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def bootstrap(y, p, k, groups=None, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap. If `groups` is given, resample whole groups (minutes)."""
    rng = np.random.default_rng(seed)
    point = metrics_from_cm(cm_of(y, p, k))
    samples = {m: np.empty(n_boot) for m in point}
    if groups is None:
        n = len(y)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            for m, v in metrics_from_cm(cm_of(y[idx], p[idx], k)).items():
                samples[m][b] = v
    else:
        # per-group confusion matrices, then sum resampled groups
        codes, inv = np.unique(groups, return_inverse=True)
        per = np.zeros((len(codes), k * k))
        np.add.at(per, inv, np.eye(k * k)[y * k + p])
        for b in range(n_boot):
            pick = rng.integers(0, len(codes), len(codes))
            cm = per[pick].sum(0).reshape(k, k)
            for m, v in metrics_from_cm(cm).items():
                samples[m][b] = v
    return {m: {"point": round(float(point[m]), 4),
                "lo": round(float(np.percentile(samples[m], 2.5)), 4),
                "hi": round(float(np.percentile(samples[m], 97.5)), 4)} for m in point}


# --------------------------------------------------------------------------- #
def ensemble():
    return VotingClassifier([
        ("xgb", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                              eval_metric="logloss", random_state=SEED, n_jobs=-1)),
        ("lgb", LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                               random_state=SEED, verbose=-1)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED, n_jobs=-1)),
    ], voting="soft")


def multi_xgb(k):
    return XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                         objective="multi:softprob", num_class=k, eval_metric="mlogloss",
                         random_state=SEED, n_jobs=-1)


def load():
    raw = pd.read_csv(DATA_PATH)
    raw["Attack-New"] = raw["Attack"].map(ATTACK_MAP)
    raw["row_id"] = np.arange(len(raw))  # original CSV order = EVDetect's indexing
    # same rolling features as the app; row_id rides along through the time sort
    feat = engineer(raw.drop(columns=["Attack-New"])).set_index("row_id")
    raw = raw.set_index("row_id")
    raw["minute"] = pd.to_datetime(raw["time"]).dt.to_period("min").astype(str)
    feat_cols = [c for c in feat.columns if c not in METADATA]
    return raw, feat, feat_cols


def run():
    t0 = time.time()
    raw, feat, feat_cols = load()
    has_feat = raw.index.isin(feat.index)
    classes_multi = ["Backdoor", "Benign", "Cryptojacking", "DoS", "Recon"]
    results = {"meta": {
        "dataset": DATA_PATH.name, "rows": int(len(raw)), "n_boot": N_BOOT, "seed": SEED,
        "rows_without_rolling_features": int((~has_feat).sum()),
        "evdetect_published": {f"{t}|{s}": v for (t, s), v in EVD_PUBLISHED.items()},
    }, "runs": []}

    for task in ("binary", "multiclass"):
        for scope in ("full", "charging"):
            base = raw if scope == "full" else raw[raw["State"] == "charging"]
            if task == "binary":
                y_all = (base["Label"] == "attack").astype(int).to_numpy()
                k, names = 2, ["Benign", "Attack"]
            else:
                y_all = base["Attack-New"].map({c: i for i, c in enumerate(classes_multi)}).to_numpy()
                k, names = 5, classes_multi
            y_ser = pd.Series(y_all, index=base.index)

            for protocol in ("evdetect", "block"):
                if protocol == "evdetect":
                    tr_idx, te_idx = train_test_split(base.index.to_numpy(), test_size=0.2, random_state=SEED,
                                                      stratify=y_all, shuffle=True)
                    tr_idx, _val = train_test_split(tr_idx, test_size=0.2, random_state=SEED,
                                                    stratify=y_ser.loc[tr_idx], shuffle=True)
                else:
                    mins = np.asarray(raw["minute"].unique(), dtype=object)
                    tr_m, _ = train_test_split(mins, test_size=0.2, random_state=SEED)
                    in_tr = base["minute"].isin(set(tr_m))
                    tr_idx, te_idx = base.index[in_tr].to_numpy(), base.index[~in_tr].to_numpy()

                print(f"[{time.time()-t0:6.0f}s] {task:<10} {scope:<8} {protocol:<8} train={len(tr_idx)} test={len(te_idx)}", flush=True)
                y_tr, y_te_full = y_ser.loc[tr_idx].to_numpy(), y_ser.loc[te_idx].to_numpy()
                # common evaluation rows: test rows that also have rolling features
                te_common = te_idx[np.isin(te_idx, feat.index)]
                tr_feat = tr_idx[np.isin(tr_idx, feat.index)]
                y_te = y_ser.loc[te_common].to_numpy()
                groups = raw.loc[te_common, "minute"].to_numpy() if protocol == "block" else None

                preds = {}
                # --- EVDetect baselines on raw features
                Xtr, Xte = raw.loc[tr_idx, RAW], raw.loc[te_common, RAW]
                rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **EVD_RF[(task, scope)]).fit(Xtr, y_tr)
                preds["EVDetect RF"] = rf.predict(Xte)
                xg = XGBClassifier(random_state=SEED, n_jobs=-1,
                                   eval_metric="logloss" if k == 2 else "mlogloss",
                                   **EVD_XGB[(task, scope)]).fit(Xtr, y_tr)
                preds["EVDetect XGBoost"] = xg.predict(Xte)
                # reproduction check on EVDetect's full test set (protocol A only)
                repro = None
                if protocol == "evdetect":
                    Xfull = raw.loc[te_idx, RAW]
                    repro = {
                        "Random Forest": bootstrap(y_te_full, rf.predict(Xfull), k, n_boot=200),
                        "XGBoost": bootstrap(y_te_full, xg.predict(Xfull), k, n_boot=200),
                        "n_test": int(len(te_idx)),
                    }

                # --- Ours: raw-feature ablation and full pipeline
                yf_tr = y_ser.loc[tr_feat].to_numpy()
                m = ensemble() if k == 2 else multi_xgb(k)
                preds["Ours (raw features)"] = m.fit(raw.loc[tr_feat, RAW], yf_tr).predict(raw.loc[te_common, RAW])

                Ftr, Fte = feat.loc[tr_feat, feat_cols].copy(), feat.loc[te_common, feat_cols].copy()
                iso = IsolationForest(contamination=0.10, random_state=SEED, n_jobs=-1).fit(Ftr[RAW])
                Ftr["unsupervised_anomaly_score"] = iso.decision_function(Ftr[RAW])
                Fte["unsupervised_anomaly_score"] = iso.decision_function(Fte[RAW])
                m = ensemble() if k == 2 else multi_xgb(k)
                preds["Ours (full pipeline)"] = m.fit(Ftr, yf_tr).predict(Fte)

                for name, p in preds.items():
                    p = np.asarray(p).astype(int)
                    results["runs"].append({
                        "task": task, "scope": scope, "protocol": protocol, "model": name,
                        "n_train": int(len(tr_idx)), "n_test": int(len(te_common)),
                        "classes": names,
                        "metrics": bootstrap(y_te, p, k, groups=groups),
                        "confusion": cm_of(y_te, p, k).tolist(),
                        "reproduction": repro if name.startswith("EVDetect") else None,
                    })
    results["meta"]["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / "benchmark_results.json").write_text(json.dumps(results, indent=1))
    write_markdown(results)
    print(f"done in {results['meta']['seconds']}s -> {OUT}")


def write_markdown(res):
    def cell(m):
        return f"{m['point']:.3f} [{m['lo']:.3f}-{m['hi']:.3f}]"
    lines = ["# Benchmark results vs. EVDetect", "",
             f"Generated by `run_benchmark.py` ({res['meta']['n_boot']} bootstrap resamples, seed {res['meta']['seed']}). "
             "Values are point estimate [95% CI]. Precision/recall/F1 are macro-averaged; macro recall equals balanced accuracy.", ""]
    for proto, title in (("evdetect", "Protocol A: EVDetect's random row split"),
                         ("block", "Protocol B: leakage-aware 1-minute block split")):
        lines += [f"## {title}", ""]
        for task in ("binary", "multiclass"):
            for scope in ("full", "charging"):
                rows = [r for r in res["runs"] if r["protocol"] == proto and r["task"] == task and r["scope"] == scope]
                if not rows:
                    continue
                lines += [f"### {task.title()}, {'full dataset' if scope == 'full' else 'charging state only'} "
                          f"(n_test = {rows[0]['n_test']:,})", "",
                          "| Model | Accuracy | Macro F1 | Macro precision | Macro recall |", "|---|---|---|---|---|"]
                for r in rows:
                    mm = r["metrics"]
                    lines.append(f"| {r['model']} | {cell(mm['accuracy'])} | {cell(mm['f1'])} | {cell(mm['precision'])} | {cell(mm['recall'])} |")
                if proto == "evdetect":
                    pub = res["meta"]["evdetect_published"][f"{task}|{scope}"]
                    ev = [r for r in rows if r["model"] == "EVDetect XGBoost"][0]["reproduction"]
                    lines += ["", f"EVDetect published (acc / F1): RF {pub['Random Forest'][0]:.4f} / {pub['Random Forest'][1]:.4f}, "
                              f"XGBoost {pub['XGBoost'][0]:.4f} / {pub['XGBoost'][1]:.4f}, Stacking {pub['Stacking'][0]:.4f} / {pub['Stacking'][1]:.4f}. "
                              f"Our re-run on their full test set (n = {ev['n_test']:,}): RF {ev['Random Forest']['accuracy']['point']:.4f} / {ev['Random Forest']['f1']['point']:.4f}, "
                              f"XGBoost {ev['XGBoost']['accuracy']['point']:.4f} / {ev['XGBoost']['f1']['point']:.4f}."]
                lines.append("")
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
