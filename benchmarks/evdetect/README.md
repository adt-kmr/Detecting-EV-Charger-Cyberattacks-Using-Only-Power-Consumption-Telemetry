# Benchmark: this project vs. EVDetect

This folder compares the project's pipeline with **EVDetect**, a peer-reviewed
study on the same data, under two evaluation protocols.

```bash
python benchmarks/evdetect/run_benchmark.py   # ~2 minutes; needs data/raw/EVSE-B-PowerCombined.csv
```

Outputs: [`results/RESULTS.md`](results/RESULTS.md) (all tables with 95% CIs)
and `results/benchmark_results.json` (also shown in the web app's Benchmark
section).

---

## 1. Choosing the benchmark

The user suggested [kosmas15/EVDetect](https://github.com/kosmas15/EVDetect).
Before using it, it was checked against the other published work on the
CICEVSE2024 power-consumption data.

| Candidate | Same file? | Task | Split | Code / splits public | Verdict |
|---|---|---|---|---|---|
| **EVDetect**: Lazaridis et al., *Scientific Reports* 2026, [doi:10.1038/s41598-026-67199-1](https://doi.org/10.1038/s41598-026-67199-1) | Yes, `EVSE-B-PowerCombined.csv` (115,298 rows) | Binary + 5-class, full and charging-only | Stratified random 80/20, seed 42 | Yes: notebooks, exact serialized test sets, tuned hyperparameters, bootstrap CIs (MIT) | **Used as the primary benchmark** |
| Al-Tarawneh et al., *IJACSA* 16(3), 2025 | Power data | Binary + multiclass | Not reproducible from paper (SMOTE, weighted metrics) | No | Context only: RF 95.07%, LightGBM 95.41% binary accuracy (weighted) |
| Joglekar et al., hybrid IDS, arXiv 2606.23236 (IEEE IoT-J) | Power data, plus a simulated FDIA class | 8-class | Chronological + stratified 60/20/20 | Not found | Different label set; LightGBM ~70.3% accuracy, ~70.6% macro-F1 |
| Buedi et al., original CICEVSE2024 paper (2024) | Introduces the data | Baselines are on HPC/kernel events (e.g. RF 78.87%) | | | No power-consumption ML baseline to compare with |

**Why EVDetect is the right standard:** it is peer-reviewed, uses the identical
file, publishes its exact protocol and tuned models, and reports
uncertainty (1,000-sample bootstrap CIs). It is the only candidate that allows
a like-for-like, reproducible comparison.

**Its weakness:** it uses a **random row split**. On 1 Hz time-series this
puts near-identical neighbouring seconds in both train and test (temporal
leakage, see `docs/PROJECT_GUIDE.md` section 7). So EVDetect's protocol is
reproduced exactly *and* complemented by a leakage-aware protocol.

## 2. Protocols

| | A: EVDetect protocol | B: Leakage-aware |
|---|---|---|
| Split | `train_test_split(test_size=0.2, stratify=y, random_state=42)` on the CSV order, then the same 80/20 again for train/validation (models fit on 64%) | Whole 1-minute blocks to train or test (80/20, seed 42) |
| Test set | Row-for-row identical to EVDetect's | 20% of minutes |
| CIs | Row bootstrap, 1,000 resamples (as EVDetect) | **Block bootstrap**: resample test minutes (rows inside a minute are not independent) |

Scenarios (EVDetect's): binary and 5-class (`Benign, DoS, Recon, Backdoor,
Cryptojacking`), each on the full dataset and on charging-state rows only.

## 3. Models

| Name | Features | Model |
|---|---|---|
| EVDetect RF | 4 raw signals | RandomForest with EVDetect's tuned hyperparameters per scenario |
| EVDetect XGBoost | 4 raw signals | XGBoost with EVDetect's tuned hyperparameters per scenario |
| Ours (raw features) | 4 raw signals | This project's models (soft-voting ensemble / multiclass XGBoost) |
| Ours (full pipeline) | 20-sample rolling features + Isolation Forest score (25) | Same models |

"Ours (raw features)" is an **ablation**: it shows how much of the difference
is due to the model versus the temporal features.

Evaluation rows: the 19 earliest rows of the timeline have no complete
rolling window; any of them that fall into a test set are dropped for all
models (at most 9 rows per scenario), so every model is scored on the same rows.

## 4. Reproduction check

EVDetect's models re-trained here, scored on their full test set, versus the
numbers in their paper's bootstrap notebook:

| Scenario | EVDetect XGBoost published (acc / macro-F1) | Re-run here |
|---|---|---|
| Binary, full | 0.9475 / 0.8774 | 0.9479 / 0.8777 |
| Binary, charging | 0.8731 / 0.8103 | 0.8733 / 0.8109 |
| Multiclass, full | 0.6441 / 0.6551 | 0.6454 / 0.6563 |
| Multiclass, charging | 0.8571 / 0.8422 | 0.8586 / 0.8444 |

All within ±0.002, so the baseline is faithfully reproduced.

## 5. Results (macro-F1, 95% CI)

| Scenario | Best EVDetect model | Ours (raw) | **Ours (full)** |
|---|---|---|---|
| **Protocol A (EVDetect's split)** | | | |
| Binary, full | 0.878 [0.872-0.885] | 0.837 | **0.960** [0.956-0.963] |
| Binary, charging | 0.811 [0.799-0.823] | 0.799 | **0.967** [0.962-0.973] |
| Multiclass, full | 0.656 [0.650-0.662] | 0.631 | **0.840** [0.836-0.844] |
| Multiclass, charging | 0.844 [0.834-0.854] | 0.838 | **0.979** [0.976-0.983] |
| **Protocol B (1-minute blocks)** | | | |
| Binary, full | 0.876 [0.851-0.893] | 0.852 | **0.953** [0.936-0.968] |
| Binary, charging | 0.802 [0.760-0.826] | 0.791 | **0.939** [0.897-0.970] |
| Multiclass, full | 0.652 [0.632-0.672] | 0.624 | **0.819** [0.797-0.840] |
| Multiclass, charging | 0.837 [0.810-0.854] | 0.832 | **0.954** [0.927-0.975] |

Accuracy, precision and recall for every model are in
[`results/RESULTS.md`](results/RESULTS.md).

## 6. Interpretation

1. **The project's pipeline beats EVDetect in all 8 settings**, by 8 to 18
   macro-F1 points, and the 95% CIs do not overlap in any of them.
2. **The gain comes from the temporal features, not the classifier.** On raw
   signals, this project's ensemble is level with or slightly below EVDetect's
   tuned models. Adding the rolling-window features and the Isolation Forest
   score is what lifts macro-F1 from ~0.65 to ~0.84 on the hard 5-class task.
   Single 1-second readings cannot separate quiet attacks (tcp-flood, scans) from
   idle benign; 20 seconds of context can.
3. **The gain survives the leakage-aware split.** Moving from EVDetect's
   random split to 1-minute blocks costs our pipeline only 0.7 to 2.8 points,
   so the advantage is not an artefact of rolling windows leaking across a
   random split. EVDetect's raw-feature models barely change between protocols
   (they have no windows that could leak).
4. **Limits.** Both protocols still let different minutes of the same attack
   session appear in train and test. Under this project's strict chronological
   split, which EVDetect does not evaluate, all approaches degrade sharply (see
   `docs/results.md`), so cross-session generalisation remains open. The
   comparison uses EVDetect's best single tuned models; their stacking
   ensemble scored within ±0.004 macro-F1 of their XGBoost.
