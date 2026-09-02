<div align="center">

# Detecting EV Charger Cyberattacks Using Only Power-Consumption Telemetry

**Non-invasive cyber-intrusion detection for Electric Vehicle Supply Equipment (EVSE) from physical power signals alone**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](#requirements)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange)](#getting-started)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-brightgreen)](#requirements)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](#license)

</div>

---

## Overview

Electric Vehicle Supply Equipment (EVSE) is becoming an increasingly
networked and security-sensitive component of the modern grid. This project
investigates whether **cyberattacks on EV chargers can be detected from the
electrical power-consumption telemetry alone** — with *no* network-packet or
protocol-layer visibility.

Using ~6 days of 1-second power telemetry (`shunt voltage`, `bus voltage`,
`current`, `power`) from an OCPP-enabled charger, we build gradient-boosted
and ensemble detectors to classify both **binary** (attack/benign) and
**multiclass** (`DoS` / `host-attack` / `none` / `recon`) conditions.

> ⚠️ **Central finding:** Naïve random train/test splitting on time-series data
> produces artificially perfect results (~99.9% accuracy) due to **temporal
> leakage**. Leakage-aware validation yields honest performance — binary
> detection stays strong (bal. acc. **0.95**), but strict chronological,
> out-of-distribution generalization is substantially weaker and remains an
> open challenge, especially for multiclass attack identification.

---

## Highlights

- **Content-agnostic detection** — only physical power signals used.
- **Hybrid pipeline** — XGBoost + LightGBM + RandomForest **soft-voting
  ensemble**, seeded with an unsupervised **Isolation Forest** anomaly score.
- **Temporal-validation rigor** — direct A/B comparison of chronological
  (out-of-distribution) vs. stratified-by-time-block evaluation, exposing the
  leakage pitfall.
- **Explainable** — SHAP (TreeExplainer) feature attribution for every model.
- **Reproducible** — fixed seeds, self-contained numbered notebooks, and a
  pinned dependency stack.

---

## Results at a Glance

### Binary attack detection

| Strategy | Balanced Accuracy | Attack F1 | Notes |
|----------|:---:|:---:|-------|
| Naïve random split | 0.999 | ~1.00 | Leakage-inflated, not transferable |
| **Stratified block** | **0.95** | **0.99** | Leakage-aware, honest |
| Chronological (OOD) | 0.72 | 0.84 | Future-horizon generalization |

### Multiclass attack-type identification

| Strategy | Accuracy | Macro-F1 | Behaviour |
|----------|:---:|:---:|-------|
| **Stratified block** | **0.80** | **0.81** | DoS .61 · host .95 · none .93 · recon .74 |
| Chronological (OOD) | 0.08 (bal.) | — | Degenerates to single class |

*Full analysis in [`docs/results.md`](docs/results.md).*

---

## Repository Structure

```
.
├── notebooks/                       # Self-contained, numbered experiments
│   ├── 01_naive_random_split_baseline.ipynb   # Leakage baseline (~99.9% acc)
│   ├── 02_chronological_split_method.ipynb    # Strict OOD temporal evaluation
│   └── 03_validation_strategy_comparison.ipynb# Chronological vs. block split
├── docs/                            # Background, methodology, results
│   ├── background.md
│   ├── methodology.md
│   └── results.md
├── data/                            # Dataset (raw/processed; git-ignored)
├── figures/                         # Generated plots (confusion, SHAP, ...)
├── reports/                         # Formal report artifacts
├── references/                      # Source material & literature
├── requirements.txt                 # Pinned dependency stack
└── README.md
```

---

## Getting Started

### Prerequisites

- Python **3.10+**
- [Jupyter](https://jupyter.org/) / JupyterLab

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/adt-kmr/Detecting-EV-Charger-Cyberattacks-Using-Only-Power-Consumption-Telemetry
cd Detecting-EV-Charger-Cyberattacks-Using-Only-Power-Consumption-Telemetry

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the notebooks

```bash
# Launch Jupyter, then open notebooks/ in order
jupyter lab            # or: jupyter notebook
```

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_naive_random_split_baseline.ipynb` | Demonstrates the leakage-inflated baseline |
| 02 | `02_chronological_split_method.ipynb` | Chronological (OOD) evaluation, binary + multiclass, SHAP |
| 03 | `03_validation_strategy_comparison.ipynb` | A/B test of validation strategies |

> **Note.** The dataset (`EVSE-B-PowerCombined (1).csv`) is not committed to
> this repository. Place it under `data/raw/` (or mount it via Google Colab as
> in the original notebooks) before running.

---

## Methodology Highlights

1. **Feature engineering** — rolling (`window=20`) `delta`, `mean`, `std`,
   `skew`, `crest` per physical signal.
2. **Anomaly injection** — Isolation Forest (`contamination=0.10`) score added
   as an unsupervised feature.
3. **Modeling** — XGBoost (binary + multiclass), LightGBM, RandomForest, and a
   soft-voting ensemble.
4. **Validation** — chronological/block splits vs. random split; balanced
   accuracy, precision/recall/F1, confusion matrix, ROC/PR-AUC, SHAP.

See [`docs/methodology.md`](docs/methodology.md) for the full description.

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/background.md`](docs/background.md) | Problem context, motivation, why temporal validation matters |
| [`docs/methodology.md`](docs/methodology.md) | Data, features, models, validation strategies, metrics |
| [`docs/results.md`](docs/results.md) | Full quantitative results and interpretation |

---

## Contributing

Contributions are welcome. Please read
[`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## ⚠️ Limitations & Future Work

- **Out-of-distribution generalization** is the hardest gap: multiclass
  attack-type identification collapses on a strict future horizon
  (bal. acc. ~0.08).
- Binary detection on the chronological horizon (~0.72) suggests room for
  improved temporal features, sequence models (LSTM/Transformer), or
  domain-adaptation.
- Active-learning and continual-learning strategies could help detectors adapt
  to newly emerging attack signatures.

---


## 📜 License

This project is released under the [MIT License](LICENSE).
