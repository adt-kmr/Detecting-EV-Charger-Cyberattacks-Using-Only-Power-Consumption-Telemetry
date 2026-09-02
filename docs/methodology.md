# Methodology

## Overview

Three experimental notebooks (`notebooks/`) progressively build toward an
honest, leakage-aware evaluation of cyberattack detection from EVSE power
telemetry. The notebooks are numbered in reading order and each targets a
distinct methodological question.

## Data

**Source:** EVSE Dataset B — *PowerCombined* telemetry.
Span: ~6 days (2023-12-24 → 2023-12-30), 1-second sampling.

**Raw signals per sample (10 columns):**
- `time` — timestamp
- `shunt_voltage` — voltage across the current-sense shunt (µV)
- `bus_voltage_V` — DC bus voltage (V)
- `current_mA` — load current (mA)
- `power_mW` — instantaneous power draw (mW)
- `State` — nominal charger state (e.g., `idle`)
- `Attack` — specific attack label (e.g., `syn-flood`)
- `Attack-Group` — coarse category: `DoS` / `host-attack` / `none` / `recon`
- `Label` — binary label: `attack` / `benign`
- `interface` — communication interface (e.g., `ocpp`)

**Class balance (binary):** ~87.5% attack / ~12.5% benign (highly imbalanced).

## Feature Engineering

For each physical signal, a rolling window (`window = 20`) is used to derive
time-window statistics that characterize the recent electrical behavior:

- `delta` — first difference (rate of change)
- `mean` — rolling mean (smoothed level)
- `std` — rolling standard deviation (variability / disturbance)
- `skew` — rolling skewness (asymmetry of disturbances)
- `crest` — rolling crest factor (peak-to-RMS ratio)

Additionally, an unsupervised **Isolation Forest** (contamination `0.10`)
anomaly score is used as an injection feature, capturing "atypicality" of a
sample relative to its local structure without any label supervision.

## Models

| Model | Role |
|-------|------|
| `XGBClassifier` | Primary gradient-boosted tree model (binary + multiclass via `multi:softprob`) |
| `LGBMClassifier` | Boosted-tree partner in the ensemble |
| `RandomForestClassifier` | Bagged-tree partner in the ensemble |
| **Soft-voting ensemble** | Probability-averaged combination of the three above |
| `IsolationForest` | Unsupervised anomaly scorer (feature source) |

## Validation Strategies

Two evaluation regimes are directly compared (the core scientific
contribution):

### Strategy 1 — Pure Chronological (Out-of-Distribution)
- Data ordered in time; first 80% used for training, final 20% held out.
- Held-out window contains **only `attack`** samples (no `benign`), i.e. an
  out-of-distribution future horizon. Honest and conservative, but the test set
  is class-imbalanced by construction.

### Strategy 2 — Stratified Block Split (Leakage-Aware Representation)
- Data chunked into **1-minute time blocks**.
- Blocks are randomly split 80/20, so both classes and all attack types are
  represented in both train and test — controlling temporal leakage within a
  block while preserving class coverage.

Naïve random sample split (notebook 01) is retained as a baseline illustrating
the leakage pitfall.

## Evaluation Metrics

- Accuracy & **Balanced Accuracy** (robust to imbalance)
- Precision / Recall / F1 (per class)
- Confusion matrix
- ROC-AUC and Precision-Recall AUC
- **SHAP** (TreeExplainer) global feature importance & summary plots
- Multiclass per-class F1 + macro-average (across `Attack-Group`)

## Reproducibility

All splits use fixed `random_state` values, and each notebook
self-documents its pipeline end-to-end. See [`requirements.txt`](../requirements.txt)
for the pinned dependency stack and [`README.md`](../README.md) for run
instructions.
