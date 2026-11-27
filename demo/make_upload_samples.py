"""
Create upload-ready CSV snippets for the web app's "Score your own telemetry" box.

Every snippet is cut from minutes that the default (block-split, seed 42) model
NEVER saw during training, so scoring them is a genuine unseen-data test, not a
replay of training data. Each snippet is a contiguous run of real recorded
seconds, exactly as the wattmeter logged them.

Run from the repository root (needs data/raw/EVSE-B-PowerCombined.csv):
    python demo/make_upload_samples.py
Outputs go to demo/upload_samples/ (git-ignored, like all CSVs in this repo).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pipeline import DATA_PATH, SEED, engineer  # noqa: E402

OUT = ROOT / "demo" / "upload_samples"
RAW_COLS = ["time", "shunt_voltage", "bus_voltage_V", "current_mA", "power_mW"]
LABEL_COLS = ["State", "Attack", "Attack-Group", "Label", "interface"]
MAX_ROWS = 360  # ~6 minutes of 1 Hz data per snippet


def held_out_minutes():
    """Reproduce the app's block split exactly and return the test minutes."""
    df = engineer(pd.read_csv(DATA_PATH))
    blocks = df["time"].dt.to_period("min")
    _, test_blocks = train_test_split(blocks.unique(), test_size=0.2, random_state=SEED)
    return set(test_blocks)


def contiguous_runs(raw, test_minutes):
    """Runs of consecutive held-out minutes with one attack type."""
    raw = raw.copy()
    raw["_t"] = pd.to_datetime(raw["time"])
    raw = raw.sort_values("_t", kind="stable").reset_index(drop=True)
    raw["_m"] = raw["_t"].dt.to_period("min")
    raw = raw[raw["_m"].isin(test_minutes)].reset_index(drop=True)
    # new run whenever the attack changes or a minute is skipped
    minute_gap = raw["_m"].astype("int64").diff().fillna(0) > 1
    new_run = (raw["Attack"] != raw["Attack"].shift()) | (raw["State"] != raw["State"].shift()) | minute_gap
    raw["_run"] = new_run.cumsum()
    return raw


def pick(runs, attack, state=None):
    cand = runs[(runs["Attack"] == attack) & ((runs["State"] == state) if state else True)]
    if cand.empty:
        return None
    longest = cand.groupby("_run").size().idxmax()
    return cand[cand["_run"] == longest].head(MAX_ROWS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    runs = contiguous_runs(raw, held_out_minutes())

    picks = {
        "01_benign_idle": pick(runs, "none", "idle"),
        "02_benign_charging": pick(runs, "none", "charging"),
        "03_dos_syn_flood": pick(runs, "syn-flood"),
        "04_dos_tcp_flood": pick(runs, "tcp-flood"),
        "05_recon_vuln_scan": pick(runs, "vuln-scan"),
        "06_cryptojacking_charging": pick(runs, "cryptojacking", "charging"),
        "07_backdoor_charging": pick(runs, "Backdoor", "charging"),
    }
    for name, seg in picks.items():
        if seg is None or len(seg) < 40:
            print(f"skip {name}: no long enough held-out run")
            continue
        seg[RAW_COLS + LABEL_COLS].to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv  rows={len(seg)}  {seg['time'].iloc[0]} -> {seg['time'].iloc[-1]}")

    # A "sensor feed" with NO label columns: benign idle followed by a SYN flood,
    # i.e. what a deployed wattmeter would actually deliver.
    feed = pd.concat([picks["01_benign_idle"].head(180), picks["03_dos_syn_flood"].head(180)])
    feed[RAW_COLS].to_csv(OUT / "08_unlabelled_sensor_feed.csv", index=False)
    print(f"08_unlabelled_sensor_feed.csv  rows={len(feed)}  (benign idle then syn-flood, no labels)")


if __name__ == "__main__":
    main()
