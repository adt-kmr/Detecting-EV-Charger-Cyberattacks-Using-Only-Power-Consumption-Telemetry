"""
Model pipeline for the EVSE power-telemetry intrusion detector.

Mirrors the notebooks (feature engineering, Isolation Forest score,
XGB + LGBM + RF soft-voting ensemble for binary detection and an XGBoost
multiclass model for Attack-Group), but packaged so a web server can train,
evaluate and replay predictions on demand.
"""
from __future__ import annotations

import io
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "EVSE-B-PowerCombined.csv"

SIGNALS = ["shunt_voltage", "bus_voltage_V", "current_mA", "power_mW"]
WINDOW = 20
METADATA = ["time", "State", "Attack", "Attack-Group", "Label", "interface"]
STRATEGIES = {
    "block": "Stratified 1-minute blocks (leakage-aware)",
    "chrono": "Pure chronological 80/20 (out-of-distribution)",
    "random": "Naive random row split (leakage baseline)",
}
SEED = 42


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling transient features exactly as in the notebooks."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    for col in SIGNALS:
        r = df[col].rolling(window=WINDOW)
        df[f"{col}_delta"] = df[col].diff()
        df[f"{col}_mean"] = r.mean()
        df[f"{col}_std"] = r.std()
        df[f"{col}_skew"] = r.skew()
        df[f"{col}_crest"] = df[col] / (df[f"{col}_mean"] + 1e-5)
    return df.dropna(subset=[c for c in df.columns if c not in METADATA]).reset_index(drop=True)


def make_ensemble() -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                  eval_metric="logloss", random_state=SEED, n_jobs=-1)),
            ("lgb", LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                   random_state=SEED, verbose=-1)),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=8,
                                          random_state=SEED, n_jobs=-1)),
        ],
        voting="soft",
    )


def _f(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 4)


def _per_class(y_true, y_pred, labels, names):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return [
        {"name": n, "precision": _f(p[i]), "recall": _f(r[i]), "f1": _f(f[i]), "support": int(s[i])}
        for i, n in enumerate(names)
    ]


# --------------------------------------------------------------------------- #
# Trained state
# --------------------------------------------------------------------------- #
@dataclass
class Trained:
    strategy: str
    feature_cols: list[str]
    iso: IsolationForest
    ensemble: VotingClassifier
    multi: XGBClassifier
    le_bin: LabelEncoder
    le_multi: LabelEncoder
    attack_idx: int
    test: pd.DataFrame            # test rows (chronological) with predictions attached
    metrics: dict
    segments: list[dict]


@dataclass
class Engine:
    raw: pd.DataFrame | None = None
    feat: pd.DataFrame | None = None
    trained: Trained | None = None
    status: dict = field(default_factory=lambda: {"state": "idle", "step": "", "progress": 0.0})
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ---- data ------------------------------------------------------------- #
    def load(self):
        if self.feat is None:
            self._set("loading", "Reading telemetry CSV", 0.03)
            self.raw = pd.read_csv(DATA_PATH)
            self._set("loading", "Engineering rolling transient features", 0.08)
            self.feat = engineer(self.raw)
        return self.feat

    def dataset_info(self) -> dict | None:
        df = self.feat
        if df is None:  # still being loaded by the training thread
            return None
        return {
            "rows": int(len(self.raw)),
            "rows_after_features": int(len(df)),
            "start": str(df["time"].min()),
            "end": str(df["time"].max()),
            "label_counts": df["Label"].value_counts().to_dict(),
            "group_counts": df["Attack-Group"].value_counts().to_dict(),
            "attack_counts": df["Attack"].value_counts().to_dict(),
            "signals": SIGNALS,
        }

    def _set(self, state, step, progress):
        self.status = {"state": state, "step": step, "progress": round(progress, 3),
                       "strategy": self.status.get("strategy")}

    # ---- training --------------------------------------------------------- #
    def train(self, strategy: str):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy}")
        with self.lock:
            t0 = time.time()
            self.status["strategy"] = strategy
            df = self.load()

            self._set("training", "Splitting data", 0.12)
            if strategy == "chrono":
                cut = int(len(df) * 0.8)
                train_mask = np.zeros(len(df), bool)
                train_mask[:cut] = True
            elif strategy == "block":
                blocks = df["time"].dt.to_period("min")
                ub = blocks.unique()
                tr_b, _ = train_test_split(ub, test_size=0.2, random_state=SEED)
                train_mask = blocks.isin(tr_b).to_numpy()
            else:
                idx = np.arange(len(df))
                tr_i, _ = train_test_split(idx, test_size=0.2, random_state=SEED,
                                           stratify=df["Label"])
                train_mask = np.zeros(len(df), bool)
                train_mask[tr_i] = True

            tr, te = df[train_mask].copy(), df[~train_mask].copy()

            # Isolation Forest is fit on training rows only (no label use, no test peeking)
            self._set("training", "Fitting Isolation Forest anomaly scorer", 0.18)
            iso = IsolationForest(contamination=0.10, random_state=SEED, n_jobs=-1)
            iso.fit(tr[SIGNALS])
            tr["unsupervised_anomaly_score"] = iso.decision_function(tr[SIGNALS])
            te["unsupervised_anomaly_score"] = iso.decision_function(te[SIGNALS])

            feature_cols = [c for c in df.columns if c not in METADATA] + ["unsupervised_anomaly_score"]
            le_bin = LabelEncoder().fit(df["Label"])
            le_multi = LabelEncoder().fit(df["Attack-Group"])
            attack_idx = int(np.where(le_bin.classes_ == "attack")[0][0])

            Xtr, Xte = tr[feature_cols], te[feature_cols]
            yb_tr, yb_te = le_bin.transform(tr["Label"]), le_bin.transform(te["Label"])
            ym_tr, ym_te = le_multi.transform(tr["Attack-Group"]), le_multi.transform(te["Attack-Group"])

            self._set("training", "Training XGBoost + LightGBM + RandomForest ensemble", 0.30)
            ens = make_ensemble()
            ens.fit(Xtr, yb_tr)

            self._set("training", "Training XGBoost multiclass (Attack-Group)", 0.62)
            n_cls = len(le_multi.classes_)
            present = np.unique(ym_tr)
            multi = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                  objective="multi:softprob", num_class=n_cls,
                                  eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
            # XGBoost needs every class id in training labels; chrono split may lack some.
            if len(present) < n_cls:
                missing = [c for c in range(n_cls) if c not in present]
                pad = Xtr.iloc[: len(missing)].copy()
                multi.fit(pd.concat([Xtr, pad]), np.concatenate([ym_tr, missing]),
                          sample_weight=np.concatenate([np.ones(len(Xtr)), np.full(len(missing), 1e-6)]))
            else:
                multi.fit(Xtr, ym_tr)

            self._set("training", "Evaluating on held-out test set", 0.80)
            P = ens.predict_proba(Xte)
            p_att = P[:, attack_idx]
            yhat_b = ens.predict(Xte)
            per_model = {}
            for name, est in ens.named_estimators_.items():
                pp = est.predict_proba(Xte)
                te[f"p_{name}"] = pp[:, attack_idx]
                per_model[name] = {
                    "balanced_accuracy": _f(balanced_accuracy_score(yb_te, pp.argmax(1))),
                    "accuracy": _f(accuracy_score(yb_te, pp.argmax(1))),
                }
            per_model["ensemble"] = {
                "balanced_accuracy": _f(balanced_accuracy_score(yb_te, yhat_b)),
                "accuracy": _f(accuracy_score(yb_te, yhat_b)),
            }

            Pm = multi.predict_proba(Xte)
            yhat_m = Pm.argmax(1)

            y_att = (yb_te == attack_idx).astype(int)
            both = len(np.unique(y_att)) == 2
            roc = roc_auc_score(y_att, p_att) if both else None
            if both:
                pr, rc, _ = precision_recall_curve(y_att, p_att)
                pr_auc = auc(rc, pr)
            else:
                pr_auc = None

            self._set("training", "Computing TreeSHAP attributions", 0.90)
            booster = ens.named_estimators_["xgb"].get_booster()
            samp = Xte.sample(min(2000, len(Xte)), random_state=SEED)
            contrib = booster.predict(xgb.DMatrix(samp), pred_contribs=True)[:, :-1]
            # booster output is log-odds of class 1; flip sign so + means "toward attack"
            sign = 1 if attack_idx == 1 else -1
            shap_imp = np.abs(contrib).mean(0)
            shap_dir = (contrib * sign).mean(0)
            order = np.argsort(-shap_imp)
            global_shap = [{"feature": feature_cols[i], "importance": _f(shap_imp[i]),
                            "mean_signed": _f(shap_dir[i])} for i in order]

            bin_names = list(le_bin.classes_)
            multi_names = list(le_multi.classes_)
            metrics = {
                "strategy": strategy,
                "strategy_label": STRATEGIES[strategy],
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "train_seconds": round(time.time() - t0, 1),
                "binary": {
                    "classes": bin_names,
                    "accuracy": _f(accuracy_score(yb_te, yhat_b)),
                    "balanced_accuracy": _f(balanced_accuracy_score(yb_te, yhat_b)),
                    "roc_auc": _f(roc),
                    "pr_auc": _f(pr_auc),
                    "per_class": _per_class(yb_te, yhat_b, list(range(len(bin_names))), bin_names),
                    "confusion": confusion_matrix(yb_te, yhat_b, labels=list(range(len(bin_names)))).tolist(),
                    "test_classes_present": [bin_names[i] for i in np.unique(yb_te)],
                    "per_model": per_model,
                },
                "multiclass": {
                    "classes": multi_names,
                    "accuracy": _f(accuracy_score(ym_te, yhat_m)),
                    "balanced_accuracy": _f(balanced_accuracy_score(ym_te, yhat_m)),
                    "macro_f1": _f(np.mean([c["f1"] for c in _per_class(ym_te, yhat_m, list(np.unique(ym_te)), [multi_names[i] for i in np.unique(ym_te)])])),
                    "per_class": _per_class(ym_te, yhat_m, list(range(n_cls)), multi_names),
                    "confusion": confusion_matrix(ym_te, yhat_m, labels=list(range(n_cls))).tolist(),
                },
                "global_shap": global_shap,
            }

            te["p_attack"] = p_att
            te["pred_label"] = le_bin.inverse_transform(yhat_b)
            for i, n in enumerate(multi_names):
                te[f"pm_{n}"] = Pm[:, i]
            te["pred_group"] = le_multi.inverse_transform(yhat_m)
            te = te.reset_index(drop=True)

            # contiguous episodes in the (chronological) test stream for "jump to"
            seg_id = (te["Attack"] != te["Attack"].shift()).cumsum()
            segments = []
            for _, g in te.groupby(seg_id, sort=False):
                if len(g) < 30:
                    continue
                segments.append({
                    "cursor": int(g.index[0]),
                    "length": int(len(g)),
                    "attack": str(g["Attack"].iloc[0]),
                    "group": str(g["Attack-Group"].iloc[0]),
                    "state": str(g["State"].iloc[0]),
                    "start": str(g["time"].iloc[0]),
                })

            self.trained = Trained(strategy, feature_cols, iso, ens, multi, le_bin, le_multi,
                                   attack_idx, te, metrics, segments)
            self._set("ready", "Models trained", 1.0)

    def train_async(self, strategy: str):
        def run():
            try:
                self.train(strategy)
            except Exception as e:  # surface errors to the UI
                self._set("error", f"{type(e).__name__}: {e}", 0.0)
        self.status["strategy"] = strategy
        self._set("training", "Queued", 0.01)
        threading.Thread(target=run, daemon=True).start()

    # ---- live replay ------------------------------------------------------ #
    def stream(self, cursor: int, n: int) -> dict:
        t = self.trained
        te = t.test
        n = max(1, min(n, 500))
        idx = [(cursor + i) % len(te) for i in range(n)]
        rows = te.iloc[idx]
        # per-row TreeSHAP for the last row in the batch (what the UI is "looking at")
        last = rows.iloc[[-1]][t.feature_cols]
        booster = t.ensemble.named_estimators_["xgb"].get_booster()
        c = booster.predict(xgb.DMatrix(last), pred_contribs=True)[0]
        sign = 1 if t.attack_idx == 1 else -1
        contrib = c[:-1] * sign
        top = np.argsort(-np.abs(contrib))[:8]
        explanation = [{"feature": t.feature_cols[i], "value": _f(last.iloc[0, i]),
                        "contribution": _f(contrib[i])} for i in top]
        out = []
        groups = list(t.le_multi.classes_)
        for i, rd in zip(idx, rows.to_dict("records")):
            out.append({
                "i": int(i),
                "time": str(rd["time"]),
                "signals": {s: float(rd[s]) for s in SIGNALS},
                "state": rd["State"],
                "true_label": rd["Label"],
                "true_group": rd["Attack-Group"],
                "true_attack": rd["Attack"],
                "p_attack": _f(rd["p_attack"]),
                "p_models": {"xgb": _f(rd["p_xgb"]), "lgb": _f(rd["p_lgb"]), "rf": _f(rd["p_rf"])},
                "pred_label": rd["pred_label"],
                "pred_group": rd["pred_group"],
                "p_groups": {g: _f(rd[f"pm_{g}"]) for g in groups},
                "anomaly_score": _f(rd["unsupervised_anomaly_score"]),
            })
        return {"cursor": int((cursor + n) % len(te)), "total": int(len(te)),
                "rows": out, "explanation": explanation}

    # ---- arbitrary CSV inference ----------------------------------------- #
    def predict_csv(self, content: bytes) -> dict:
        t = self.trained
        df = pd.read_csv(io.BytesIO(content))
        missing = [c for c in ["time", *SIGNALS] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        for m in METADATA:
            if m not in df.columns:
                df[m] = None
        f = engineer(df)
        if f.empty:
            raise ValueError(f"Need more than {WINDOW} rows to build rolling features")
        f["unsupervised_anomaly_score"] = t.iso.decision_function(f[SIGNALS])
        X = f[t.feature_cols]
        p_att = t.ensemble.predict_proba(X)[:, t.attack_idx]
        pred = t.le_bin.inverse_transform(t.ensemble.predict(X))
        Pm = t.multi.predict_proba(X)
        groups = t.le_multi.inverse_transform(Pm.argmax(1))
        res = {
            "rows": int(len(f)),
            "attack_fraction": _f((pred == "attack").mean()),
            "mean_p_attack": _f(p_att.mean()),
            "group_counts": pd.Series(groups).value_counts().to_dict(),
        }
        if f["Label"].notna().all():
            y = f["Label"].to_numpy()
            res["labelled"] = True
            res["accuracy"] = _f(accuracy_score(y, pred))
            res["balanced_accuracy"] = _f(balanced_accuracy_score(y, pred))
        else:
            res["labelled"] = False
        step = max(1, len(f) // 600)
        res["series"] = {
            "p_attack": [_f(v) for v in p_att[::step]],
            "power_mW": [float(v) for v in f["power_mW"].to_numpy()[::step]],
            "true": [None if v is None else str(v) for v in f["Label"].to_numpy()[::step]],
        }
        return res
