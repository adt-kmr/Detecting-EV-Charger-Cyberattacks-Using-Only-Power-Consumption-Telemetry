# EVSE Power Sentinel (web demo)

Live frontend for the project: the backend trains the real models on the
EVSE-B PowerCombined dataset and the page streams their predictions.

## Run

1. Put `EVSE-B-PowerCombined.csv` in `data/raw/`.
2. From the repository root:

```bash
pip install -r requirements.txt
python -m uvicorn app.server:app --port 8000
```

3. Open http://localhost:8000. The first training run (block split) starts
   automatically and takes roughly 20-30 s.

## What is live

- **Retrain**: pick Block / Chronological / Random split and retrain. All
  metrics, confusion matrices and SHAP importances come from that run.
- **Stream**: replays the held-out test rows in time order through the trained
  XGBoost + LightGBM + RandomForest ensemble and the multiclass XGBoost model.
  It shows per-row TreeSHAP explanations and running accuracy against ground truth.
- **Upload**: score any CSV with `time, shunt_voltage, bus_voltage_V,
  current_mA, power_mW` columns (plus optional `Label`).

## Custom 3D charger model

Drop a glTF binary at `app/static/models/ev_charger.glb` and it replaces the
built-in clay charger automatically (auto-scaled and centred).
