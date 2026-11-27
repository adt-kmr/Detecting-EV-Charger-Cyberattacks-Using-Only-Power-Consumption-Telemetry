# Project Guide: Detecting EV-Charger Cyberattacks from Power Telemetry

This guide is for someone opening the project for the first time. It explains
what problem the project solves, what is in the data, how every model and
validation method works (with the underlying classical ML concepts), what
each element of the web application means, and how the code is organised.

Read it top to bottom once; afterwards each section works as a reference.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [The data](#2-the-data)
3. [Repository map and data flow](#3-repository-map-and-data-flow)
4. [How attack vs. benign is defined](#4-how-attack-vs-benign-is-defined)
5. [Feature engineering](#5-feature-engineering)
6. [The algorithms, in detail](#6-the-algorithms-in-detail)
7. [The three validation methods (and why they matter most)](#7-the-three-validation-methods)
8. [Metrics: how results are judged](#8-metrics-how-results-are-judged)
9. [Explainability with SHAP](#9-explainability-with-shap)
10. [The web application, element by element](#10-the-web-application-element-by-element)
11. [Code walkthrough](#11-code-walkthrough)
12. [Benchmark against published work](#12-benchmark-against-published-work)
13. [Limitations and open questions](#13-limitations-and-open-questions)
14. [Glossary](#14-glossary)

---

## 1. The project in one page

**Problem.** Electric Vehicle Supply Equipment (EVSE, i.e. a charger) is a
networked computer. Attackers can flood it with traffic (DoS), scan it
(reconnaissance), or run malicious code on it (backdoor, cryptojacking).
Usually these are detected by inspecting network packets or host logs.

**Idea.** Any computation an attacker forces on the charger costs electricity.
If we watch only the charger's **power supply** (voltage, current, power), can
we tell when it is under attack, and what kind of attack? This is
*content-agnostic* detection: it needs no access to packets, logs or software,
only a cheap power sensor.

**What the project does.**

| Step | What happens |
|---|---|
| Data | ~115k one-second power readings from a real test charger (CICEVSE2024, EVSE-B), each labelled with the attack running at that moment. |
| Features | For each reading, summarise the last 20 readings (change, average, spread, asymmetry, peakiness). |
| Models | An Isolation Forest adds an unsupervised "how unusual is this" score. Then a soft-voting ensemble of **XGBoost + LightGBM + Random Forest** decides attack vs. benign, and a separate **XGBoost** model names the attack group. |
| Validation | The scientific core: three ways of splitting train/test data are compared, showing that the common "random split" produces fake ~99.9% accuracy on time-series data. |
| Explainability | SHAP values show which features pushed each decision. |
| App | A web app trains these models live and replays the test data through them, with a 3D charger whose colour reflects the model's verdict. |

**Headline results** (leakage-aware block split): binary balanced accuracy
**0.95**, attack F1 **0.99**, 4-group macro-F1 **0.81**. On a strict "future"
split, multiclass collapses (balanced accuracy ~0.08), which is the honest
open problem.

**Is this classification or clustering?** Classification (supervised: the
models learn from labelled examples). The only unsupervised part is the
Isolation Forest, which is an anomaly scorer, not a clustering algorithm.

---

## 2. The data

**Source:** CICEVSE2024 dataset (Canadian Institute for Cybersecurity,
University of New Brunswick), file `EVSE-B-PowerCombined.csv`. EVSE-B is a
Raspberry Pi 4-based charger running OCPP (the protocol chargers use to talk to
the operator's backend). An I2C wattmeter measured its supply while scripted
attacks were run against it.

**Size:** 115,298 rows, ~6 days (2023-12-24 to 2023-12-30). Samples are ~1 per
second, but the `time` column only has **minute resolution** (all ~60 samples
in a minute share one timestamp; 1,995 distinct minutes).

### Columns

| Column | Meaning | Typical value |
|---|---|---|
| `time` | Timestamp (minute precision) | `12/25/2023 22:35` |
| `shunt_voltage` | Voltage across the wattmeter's current-sense resistor. Proportional to current. | 417 to 1214 |
| `bus_voltage_V` | Supply (bus) voltage in volts. Very stable. | 5.15 to 5.21 V |
| `current_mA` | Current drawn by the charger, milliamps | 417 to 1220 mA |
| `power_mW` | Power drawn = voltage x current, milliwatts | 2160 to 6300 mW |
| `State` | What the charger was doing: `idle` or `charging` | |
| `Attack` | Specific attack running | `syn-flood`, `none`, ... |
| `Attack-Group` | Coarse family | `DoS`, `recon`, `host-attack`, `none` |
| `Label` | Binary truth | `attack` / `benign` |
| `interface` | Interface attacked | always `ocpp` |

Only the four physical columns are model inputs. `State`, `Attack`,
`Attack-Group`, `Label`, `interface`, `time` are **metadata**; using them as
inputs would be cheating (they are the answer, or leak it).

### Classes and counts

| Attack | Group | What it is | idle rows | charging rows | Median current |
|---|---|---|---:|---:|---:|
| `none` | none (benign) | Normal operation | 7,416 | 6,947 | 520 mA |
| `syn-flood` | DoS | Floods TCP SYN packets | 13,517 | 0 | 963 mA |
| `tcp-flood` | DoS | Floods TCP connections | 21,367 | 0 | 468 mA |
| `vuln-scan` | recon | Vulnerability scanning | 26,246 | 0 | 468 mA |
| `syn-stealth` | recon | Stealth port scan | 7,072 | 0 | 469 mA |
| `Backdoor` | host-attack | Remote-control malware | 6,965 | 14,172 | 626 mA |
| `cryptojacking` | host-attack | Hidden crypto-mining | 3,122 | 8,474 | 945 mA |

Things worth noticing before any modelling:

- **Imbalance:** 87.5% of rows are attacks, only 12.5% benign. A model that
  always says "attack" gets 87.5% accuracy while being useless. This is why
  the project reports *balanced* accuracy (section 8).
- **Some attacks are electrically loud, some are quiet.** syn-flood and
  cryptojacking burn CPU, so current jumps to ~950 mA. tcp-flood, vuln-scan and
  syn-stealth sit at ~468 mA, which is right inside the benign idle range.
  These quiet attacks are the hard part.
- **DoS and recon only happen while idle; host attacks happen in both states.**
  The charging state also raises current, so "high current" can mean either
  "charging" or "attack". The model never sees `State`, so it must work this
  out from the signal shape.

---

## 3. Repository map and data flow

```
.
├── notebooks/                       Research experiments (run in order)
│   ├── 01_naive_random_split_baseline.ipynb   Shows the leakage trap (~99.9%)
│   ├── 02_chronological_split_method.ipynb    Full pipeline, strict future split
│   └── 03_validation_strategy_comparison.ipynb Chronological vs. block split
├── app/                             Live web application
│   ├── pipeline.py                  Features, training, evaluation, replay, CSV scoring
│   ├── server.py                    FastAPI web server (HTTP API + static files)
│   └── static/                      Frontend
│       ├── index.html               Page layout
│       ├── styles.css               Clay visual style
│       ├── main.js                  Talks to the API, draws charts, animations
│       └── charger.js               three.js 3D charger scene
├── benchmarks/evdetect/             Comparison with the EVDetect paper
├── docs/                            Background, methodology, results, this guide
├── data/raw/                        Dataset goes here (git-ignored)
└── requirements.txt
```

Data flow when you start the app:

```
 EVSE-B-PowerCombined.csv
          │  read, sort by time
          ▼
 engineer(): 20-sample rolling features  ──►  25-column feature table
          │
          │  split rows into train / test (block, chronological or random)
          ▼
 ┌──────── train rows ────────┐            ┌──── test rows ────┐
 │ IsolationForest.fit        │──score──►  │ anomaly score     │
 │ VotingClassifier.fit       │            │ P(attack)         │
 │   (XGB + LGBM + RF)        │──predict─► │ predicted label   │
 │ XGBoost multiclass.fit     │            │ group probs       │
 └────────────────────────────┘            └─────────┬─────────┘
                                                     │ metrics, confusion matrices,
                                                     │ global SHAP (computed once)
                                                     ▼
                                   browser polls /api/stream and replays
                                   test rows in time order + per-row SHAP
```

---

## 4. How attack vs. benign is defined

There are two separate notions; keep them apart.

**Ground truth (what really happened).** Comes from the dataset: a row is
`benign` when the experimenters were running no attack (`Attack == none`),
otherwise `attack`. The models never see this at prediction time; it is only
used for training and for scoring.

**Prediction (what the model thinks).**

1. Each of the three ensemble members outputs a probability that the row is an
   attack: `p_xgb`, `p_lgb`, `p_rf`.
2. Soft voting averages them: `P(attack) = (p_xgb + p_lgb + p_rf) / 3`.
3. **Decision rule:** `Attack` if `P(attack) >= 0.5`, else `Benign`.

So "attack" means *the average of three tree models is at least 50% confident
the current power pattern looks like the attack examples it was trained on*.

The attack *group* (DoS / host-attack / none / recon) is predicted by a
**different, independent** model (multiclass XGBoost): it outputs four
probabilities that sum to 1 and the largest one wins. Because the two models
are independent they can occasionally disagree (e.g. binary says `Benign` while
the group model's top choice is `DoS`). The binary verdict is the one used for
alerting.

The 0.5 threshold is a default, not a law. Raising it gives fewer false alarms
but misses more attacks; lowering it does the opposite. The ROC and PR curves
(section 8) describe that trade-off across all thresholds.

---

## 5. Feature engineering

A single reading (e.g. "520 mA") says little: benign charging and a backdoor
can both draw 520 mA. What differs is **behaviour over time**: floods cause
spiky, jittery current; mining causes a flat high plateau; scans cause small
periodic bumps. So for every reading we summarise the previous 20 readings
(about 20 seconds). Windows look **backwards only**, so a feature never uses
future data.

For each of the 4 signals `x` (`shunt_voltage`, `bus_voltage_V`, `current_mA`,
`power_mW`), with `x_t` the current reading:

| Feature | Formula | What it captures |
|---|---|---|
| raw | `x_t` | Level right now |
| `_delta` | `x_t - x_{t-1}` | Sudden jumps (rate of change) |
| `_mean` | average of `x_{t-19..t}` | Smoothed level (plateaus such as mining) |
| `_std` | standard deviation of the window | Jitter / instability (floods) |
| `_skew` | skewness of the window | Asymmetry: occasional spikes up vs. dips down |
| `_crest` | `x_t / (mean + 1e-5)` | How far the current value sticks out from its recent average |

That is 4 raw + 4 x 5 engineered = **24 features**, plus the Isolation Forest
anomaly score = **25 model inputs**. The first 19 rows of the file have no
full window and are dropped.

Statistical refresher:

- **Standard deviation** `sqrt(mean((x - mean)^2))`: typical distance from the
  mean.
- **Skewness** `mean((x - mean)^3) / std^3`: 0 for a symmetric window; positive
  when there is a long tail of high values (occasional spikes up).
- **Crest factor** in signal processing is peak / RMS; here it is approximated as
  current value / rolling mean.

The SHAP importance chart in the app shows which of these the model relies on
most. In practice `bus_voltage_V_mean` (the supply sags slightly under load),
the anomaly score and the shunt-voltage spread rank highest.

---

## 6. The algorithms, in detail

All the supervised models here are built from **decision trees**, so we start
there and build up.

### 6.1 Decision tree (the building block)

A decision tree asks a sequence of yes/no questions about the features, for
example *"is current_mA_std > 35?"*, and each leaf holds a prediction.

**How it is learned (CART algorithm).** Start with all training rows in one
node. Try every feature and many thresholds; pick the split that makes the two
child nodes as *pure* as possible, then repeat on each child until a stopping
rule (max depth, minimum samples) is hit.

Purity is measured with **Gini impurity** `G = 1 - sum_k p_k^2` (where `p_k` is
the share of class `k` in the node) or **entropy** `H = -sum_k p_k log p_k`. A
node that is 100% attack has `G = 0`. The chosen split maximises the drop in
impurity, weighted by node sizes.

**Strengths:** handles non-linear patterns and feature interactions, needs no
feature scaling, readable. **Weakness:** a single deep tree memorises noise
(high **variance**: small changes in data give a very different tree). The two
fixes are *bagging* (Random Forest) and *boosting* (XGBoost, LightGBM).

### 6.2 Bias, variance and ensembles (the classical idea behind everything)

The error of a model on new data splits into roughly:

- **Bias:** the model is too simple to capture the pattern (underfitting).
- **Variance:** the model is too sensitive to the specific training sample
  (overfitting).
- **Noise:** randomness no model can remove.

Ensembles combine many models to cut one of these:

| Family | How | Mainly reduces | Example |
|---|---|---|---|
| Bagging | Train many deep trees on random resamples, average them | Variance | Random Forest |
| Boosting | Train shallow trees one after another, each fixing the previous errors | Bias (and variance through regularisation) | XGBoost, LightGBM |
| Voting / Stacking | Combine *different* model types | Both, via diversity | This project's ensemble, EVDetect's stacking |

### 6.3 Random Forest

1. Draw a **bootstrap sample** of the training rows (sample *n* rows with
   replacement; ~63% unique rows per tree).
2. Grow a decision tree on it, but at every split only consider a **random
   subset of features** (e.g. `sqrt(25) ≈ 5`). This decorrelates the trees.
3. Repeat for many trees (here 100, depth up to 8).
4. Predict: average the trees' class probabilities.

Averaging many decorrelated, individually noisy trees cancels out their
individual mistakes. If the trees' errors had correlation `rho` and variance
`sigma^2`, the forest's variance is about `rho*sigma^2 + (1-rho)*sigma^2/B`,
which is why random feature selection (lower `rho`) matters.

Project settings: `n_estimators=100, max_depth=8`. The depth cap keeps it
fast and a little more conservative than the boosters, which adds diversity to
the ensemble.

### 6.4 Gradient boosting (the idea behind XGBoost and LightGBM)

Boosting builds an **additive model**: `F(x) = f_1(x) + f_2(x) + ... + f_M(x)`,
where each `f_m` is a small tree. Training is sequential:

1. Start with a constant prediction (e.g. the log-odds of the attack rate).
2. Compute, for every training row, how the loss would change if the
   prediction moved: the **gradient** of the loss. For log-loss this is simply
   `p - y` (predicted probability minus truth), i.e. "how wrong are we, and in
   which direction".
3. Fit a small tree to predict those gradients (to the negative gradient).
4. Add it to the model, scaled by a **learning rate** `eta` (here 0.05):
   `F <- F + eta * f_m`. Small steps generalise better.
5. Repeat for `M` rounds (here 100).

This is gradient descent, but in "function space": each tree is one step
towards lower loss. For classification the model outputs a score (log-odds);
the probability is `sigmoid(F(x)) = 1 / (1 + e^(-F(x)))`.

### 6.5 XGBoost (eXtreme Gradient Boosting)

XGBoost is gradient boosting with three key upgrades:

- **Second-order optimisation.** It uses both the gradient `g` and the
  curvature (Hessian) `h` of the loss, a Newton step instead of a plain
  gradient step. The best value for a leaf is `w* = -sum(g) / (sum(h) + lambda)`.
- **Regularised objective.** Loss + `gamma * (number of leaves)` +
  `0.5 * lambda * sum(w^2)`. A split is only kept if its gain exceeds `gamma`,
  which prunes useless branches and controls overfitting.
- **Engineering.** Histogram-based split finding, sparsity handling, parallel
  tree construction, and column/row subsampling.

Trees grow **level-wise** (all nodes at one depth are split before going
deeper), to `max_depth` (here 5).

Used twice in this project:

- **Binary member of the ensemble** (`objective=binary:logistic`).
- **Multiclass group model** (`objective=multi:softprob`). For `K` classes it
  grows `K` trees per round (one score per class) and turns the scores into
  probabilities with **softmax**: `p_k = e^(F_k) / sum_j e^(F_j)`. That is why
  the four group probabilities in the app always sum to 100%.

### 6.6 LightGBM (Light Gradient Boosting Machine)

The same boosting idea, engineered for speed:

- **Leaf-wise growth.** Instead of splitting a whole level, it always splits
  the single leaf with the largest loss reduction. Fewer splits reach the
  same loss, but it can overfit, so `max_depth=5` caps it here.
- **Histogram binning** of features into ~255 bins makes split search cheap.
- **GOSS** (Gradient-based One-Side Sampling): keeps all rows with large
  gradients (badly predicted) and a random sample of the rest.
- **EFB** (Exclusive Feature Bundling): merges features that are rarely
  non-zero together.

### 6.7 XGBoost vs. LightGBM vs. Random Forest

| | Random Forest | XGBoost | LightGBM |
|---|---|---|---|
| Family | Bagging | Boosting | Boosting |
| Trees built | Independently, in parallel | Sequentially | Sequentially |
| Typical tree | Deep, unpruned | Shallow, level-wise | Shallow, leaf-wise |
| Main error fixed | Variance | Bias | Bias |
| Overfitting risk | Low | Medium (regularised) | Medium-higher (leaf-wise) |
| Speed | Good | Good | Fastest |
| Probabilities | Averaged votes, tend to be squashed toward 0.5 | Sharper | Sharper |
| Result here (block split, bal. acc.) | 0.93 | 0.96 | 0.96 |

The two boosters are near-identical on this data; Random Forest is a little
weaker but makes *different* mistakes, which is what an ensemble wants.

### 6.8 Soft voting (how the three are combined)

- **Hard voting:** each model votes for a class; majority wins.
- **Soft voting (used here):** average the predicted probabilities, then
  apply the 0.5 threshold. It uses each model's confidence, not just its vote.
- **Stacking (used by EVDetect):** train a *meta-model* (e.g. logistic
  regression) on the base models' outputs to learn how to weight them. More
  flexible, but needs extra cross-validation and can overfit.

On this data the soft-voting ensemble (0.952) is about as good as the best
single booster (0.956). Ensembles rarely beat a well-tuned booster by much on
clean tabular data; their value is robustness: no single model's quirks
dominate.

### 6.9 Isolation Forest (the unsupervised part)

Goal: score how *unusual* a reading is, without using labels.

Idea: anomalies are "few and different", so they are easy to isolate.

1. Build many random trees: at each node pick a random feature and a random
   split value between that feature's min and max; keep splitting until each
   point is alone.
2. An unusual point gets isolated after few splits (short path from the root);
   a typical point needs many.
3. Average path length `E[h(x)]` over 100 trees, normalised by `c(n)` (the
   expected path length in a random binary tree of `n` points): the anomaly
   score is `s(x) = 2^(-E[h(x)] / c(n))`, close to 1 for anomalies.
4. `contamination=0.10` tells it to place the decision threshold so ~10% of
   training points count as anomalous. sklearn's `decision_function` returns a
   shifted score: **negative = anomalous, positive = normal**.

In this project it is fitted on the **training rows' raw signals** only, and
its score becomes the 25th feature for the supervised models. It contributes
a label-free sense of "this looks off", and SHAP shows the classifiers lean on
it heavily.

### 6.10 Why not logistic regression, SVM, KNN or a neural network?

The EVDetect paper (section 12) tested these on the same data:

| Model | How it decides | On this data |
|---|---|---|
| Logistic regression | A single weighted sum of features through a sigmoid: a straight-line boundary | Too simple: macro-F1 ~0.47 on the full dataset |
| SVM (RBF) | Finds a maximum-margin boundary in a kernel space | Decent but slow on 100k rows; needs scaling |
| KNN | Copies the labels of the nearest training points | Good under random splits (near-duplicates help it), sensitive to scale |
| Neural network (MLP) | Layers of weighted sums + non-linearities | Competitive but needs scaling and tuning; no advantage on 4 to 25 tabular features |
| Tree ensembles | Many axis-aligned splits combined | Best: handle mixed scales, interactions and imbalance with little tuning |

This matches the general finding that gradient-boosted trees are the strongest
default on small-to-medium tabular data.

### 6.11 Handling class imbalance

87.5% attack / 12.5% benign. Options:

- **Do nothing, but measure correctly** (balanced accuracy, macro-F1). This
  project does this.
- **Class weights:** make each benign error cost more during training.
- **Resampling / SMOTE:** create synthetic minority samples by interpolating
  between neighbours (EVDetect tried this; it did not help much).
- **Threshold tuning:** move the 0.5 cut-off.

---

## 7. The three validation methods

This is the scientific heart of the project. A model is only as trustworthy as
the way its test score was measured.

### 7.1 The key concept: leakage and autocorrelation

The test set must contain information the model has **never seen**, otherwise
the score measures memory, not skill.

Time-series rows are **autocorrelated**: second `t` looks almost exactly like
second `t+1`. Worse, the rolling features of row `t` and row `t+1` share 19
of their 20 input readings. If row `t` lands in training and row `t+1` lands in
testing, the model has effectively already seen the test row. That is **temporal
leakage**.

### 7.2 Method 1: Naive random split (notebook 01)

Shuffle all rows and put a random 20% in the test set.

- Result: **~99.9% accuracy**.
- Why it is wrong: almost every test row has its near-twin neighbours in
  training. The model just recognises the session.
- Kept in the project as a **cautionary baseline**. It is also how many
  published papers on this dataset evaluate, which is why their numbers look
  so high (section 12).

### 7.3 Method 2: Pure chronological split (notebook 02)

Sort by time. First 80% is training, last 20% is testing.

- Most realistic: you train on the past and deploy into the future.
- Problem specific to this dataset: the experiments were run in blocks, so
  the last 20% of the timeline contains **only attack rows**, and not all
  attack types. Benign is absent from the test set, so the test asks "can a
  model trained on earlier sessions recognise later, differently-recorded
  sessions?" (an **out-of-distribution** test).
- Result: binary ~0.66 to 0.72 (it is effectively attack recall); multiclass
  collapses to ~0.08 balanced accuracy, predicting nearly everything as one
  class.
- Lesson: the models have not learned a universal "signature" of each attack
  that survives a change of session. This is the honest open problem.

### 7.4 Method 3: Stratified 1-minute block split (notebook 03, app default)

Cut the timeline into 1-minute blocks (~60 rows each). Randomly assign whole
blocks to train (80%) or test (20%).

- Leakage now only happens at block edges (the first ~19 rows of a test block
  have windows that reach back into the previous minute), not inside every
  test row.
- Both classes and all attack types appear in train and test, so per-class
  scores are measurable.
- Result: binary balanced accuracy **0.95**, multiclass macro-F1 **0.81**.
- Caveat: an attack session lasts many minutes, so neighbouring minutes of
  the *same session* can be in train and test. Block splitting is
  **leakage-reduced, not leakage-free**. A stricter version would leave a gap
  ("purge") between train and test blocks, or hold out whole sessions.

### 7.5 Side by side

| | Random rows | Chronological | 1-minute blocks |
|---|---|---|---|
| Leakage | Severe | None | Small (block edges, same session) |
| Test classes | All | Attack only | All |
| Binary bal. acc. | ~0.999 | ~0.66 to 0.72 | 0.95 |
| Multiclass | (not run) | ~0.08 bal. acc. | 0.81 macro-F1 |
| Question it answers | "Can it recognise rows it nearly saw?" | "Does it transfer to new sessions?" | "Can it recognise unseen minutes of known behaviour?" |

You can reproduce all three in the app with the strategy switch.

---

## 8. Metrics: how results are judged

Start from the **confusion matrix** (rows = truth, columns = prediction). For
the attack class:

|  | Predicted attack | Predicted benign |
|---|---|---|
| **Truly attack** | TP (caught) | FN (missed attack) |
| **Truly benign** | FP (false alarm) | TN (correctly quiet) |

| Metric | Formula | Plain meaning |
|---|---|---|
| Accuracy | (TP+TN) / all | Share of rows right. Misleading under imbalance. |
| Precision | TP / (TP+FP) | When it alarms, how often is it right? |
| Recall (sensitivity, detection rate) | TP / (TP+FN) | Of real attacks, how many are caught? |
| F1 | 2·P·R / (P+R) | Balance of precision and recall for one class |
| Balanced accuracy | mean of per-class recall | Accuracy where each class counts equally. 0.5 = coin flip for 2 classes. |
| Macro-F1 | mean of per-class F1 | Treats rare classes as equally important |
| Weighted-F1 | per-class F1 weighted by support | Dominated by big classes (inflates results) |
| False alarm rate | FP / (FP+TN) | Share of benign rows wrongly flagged |
| ROC-AUC | Area under true-positive vs. false-positive rate over all thresholds | Probability a random attack scores higher than a random benign row. 0.5 = random. |
| PR-AUC | Area under precision vs. recall | Like ROC-AUC but sensitive to imbalance; more honest for rare classes |

Why the project leads with **balanced accuracy** and **macro-F1**: with 87.5%
attacks, a model that always says "attack" scores 87.5% accuracy but only 50%
balanced accuracy and ~47% macro-F1, exposing it immediately.

**Confidence intervals** (used in the benchmark): resample the test set with
replacement 1,000 times, recompute the metric, and report the 2.5th to 97.5th
percentile range. If two models' intervals overlap heavily, their difference
may be noise.

---

## 9. Explainability with SHAP

Tree ensembles are accurate but not self-explanatory. **SHAP** (SHapley
Additive exPlanations) assigns each feature a contribution to one specific
prediction, based on **Shapley values** from cooperative game theory: a
feature's credit is its average marginal effect over all possible orders in
which features could be "added" to the model.

Key property, **additivity**:

```
model output for this row = base value (average output) + sum of all feature contributions
```

For XGBoost the output is in **log-odds** (so +1.0 multiplies the odds of
attack by e ≈ 2.7). **TreeSHAP** computes exact Shapley values for tree models
in polynomial time; XGBoost has it built in (`pred_contribs=True`), which the
app uses.

- **Local explanation** (per row): which features pushed *this* decision toward
  attack (+) or benign (-).
- **Global importance:** the mean absolute contribution of each feature over
  many rows: how much the model relies on it overall.

Caveat: the app explains the **XGBoost member**, not the averaged ensemble.
It is a faithful view of one-third of the vote, and in practice the members
agree closely.

---

## 10. The web application, element by element

Start it from the repository root and open http://localhost:8000:

```
python -m uvicorn app.server:app --port 8000
```

### What "live" means

On startup the server trains every model from scratch on the real dataset
(~20-30 s). It then scores all test rows in one batch and keeps the results in
memory. The **stream replays those test rows in time order**, revealing the
real model outputs one by one (and computing a fresh SHAP explanation for
each displayed row). Nothing is hardcoded; retraining with another split
changes every number. Uploaded CSVs are scored on the spot.

### Header

| Element | Meaning |
|---|---|
| **EVSE Power Sentinel** | App title. |
| **Block split / Chronological / Random split** | Chooses the validation method (section 7) used for the next training run. |
| **⟲ Retrain models** | Retrains everything with the selected split. A progress bar shows each stage: reading data, features, Isolation Forest, ensemble, multiclass model, evaluation, SHAP. |
| **Status pill** | Green = models ready (shows which split), yellow = training, red = error / backend offline. |

### 3D charger stage

The 3D model is a visual readout of the model output. Drag to rotate.

| Element | Meaning |
|---|---|
| Clock tag (◷) | Timestamp of the row being shown. |
| "Charger: idle / charging" | The true `State` of that row (metadata; *not* a model input). |
| On-charger screen | Verdict, P(attack), predicted group and power in watts. |
| LED bar and halo rings | Purple = benign; blend toward **yellow** as P(attack) rises; they pulse and spin faster when the verdict is Attack. |
| Slight shake | Appears when P(attack) > 0.5. |
| Ground ring pulse | One pulse per scored sample, coloured by verdict. |
| Particles on the cable | Speed is proportional to the measured current. |

### Stream controls

| Control | Meaning |
|---|---|
| ▶ Play / ❚❚ Pause | Start or stop replaying test rows. |
| ▸ Step | Advance one row. |
| 1× 4× 12× 40× | Rows fetched per tick (a tick is ~0.45 s). |
| Jump to an episode | Jumps to the start of a contiguous run (≥30 rows) of one attack type in the test stream, e.g. "DoS / syn-flood". Useful to watch the model react to a specific attack. |

### Live verdict

| Element | Meaning |
|---|---|
| **Gauge (P(attack))** | Ensemble probability of attack, 0-100. Yellow arc when ≥ 50 (Attack verdict), purple when below. |
| **Attack / Benign** | The binary verdict (threshold 0.5). |
| **Predicted attack group** | Top class of the separate multiclass model. |
| **✓ / ✕ Ground truth** | The true label, group and specific attack. ✓ when the binary verdict matches the truth, ✕ when it does not. |
| "test row N of M" | Position in the replayed held-out test set. |

### Multiclass attack-group probabilities

Four bars from the multiclass XGBoost (softmax, so they sum to 100%). The
highlighted name is the predicted group. Yellow bars are attack groups,
purple is `none` (benign).

### Ensemble members

| Bar | Meaning |
|---|---|
| XGBoost / LightGBM / RandomForest | Each member's own P(attack) for this row. Yellow if that member alone would say Attack. |
| Ensemble | Their average, the number actually used. |
| Isolation Forest | The unsupervised score (number on the right: positive = normal, negative = anomalous). The bar shows "anomalousness" (longer = more unusual). |

### Live power telemetry

Four charts of the last 240 rows of the raw signals: power (mW), current (mA),
bus voltage (V), shunt voltage. The value on the right is the latest reading.
**Yellow background stripes** mark rows the model flagged as attack, so you
can see what the signal looked like when it alarmed.

### Detector output vs. ground truth

- **Purple line:** ensemble P(attack) over the last 240 rows.
- **Dashed grey line:** the 0.5 decision threshold.
- **Bottom band:** the *true* label (yellow = attack, purple = benign).
- **Orange dots:** rows where the verdict was wrong.

Below it, running statistics over everything streamed since the last reset:

| Stat | Formula |
|---|---|
| Samples scored | Rows shown so far |
| Running accuracy | correct / scored |
| Attack recall | caught attacks / true attacks |
| False alarm rate | benign rows flagged as attack / true benign rows ("n/a" until a benign row is seen) |

### Why this decision? (per-sample TreeSHAP)

The 8 features with the largest SHAP contribution for the row on screen.
Bars grow right (**yellow**, toward attack) or left (**purple**, toward
benign) from the centre line; the number is the contribution in log-odds.
Hover a name to see the feature's value. Example reading: "+2.15
`unsupervised_anomaly_score`" means the Isolation Forest score alone raised the
odds of attack by about e^2.15 ≈ 8.6x.

### Global feature importance (mean |SHAP|)

The 10 features the XGBoost member relies on most, averaged over 2,000
held-out rows after training. Yellow if, on average, the feature pushes toward
attack. This is the app version of the notebooks' SHAP summary plot.

### Held-out evaluation

Computed once on the full test set right after training (subtitle shows split,
train/test sizes and training time).

| KPI | Meaning |
|---|---|
| Balanced accuracy | Binary ensemble, mean of attack recall and benign recall |
| Attack F1 | F1 of the attack class; with ROC-AUC and PR-AUC beneath |
| Multiclass accuracy | Share of rows whose group was predicted correctly (balanced accuracy beneath) |
| Macro F1 (groups) | Mean F1 across the four groups |

With the chronological split a yellow note appears: the test window contains
only attacks, so ROC/PR-AUC are undefined and balanced accuracy equals recall
on that single class.

### Confusion matrices and model comparison

- **Binary / Attack-group confusion matrix:** rows = truth, columns =
  prediction. Purple diagonal = correct; yellow off-diagonal = errors (darker =
  more rows). Hover a cell for the row percentage. Example: the DoS → recon
  cell is large because tcp-flood (DoS) and the scans (recon) all sit at
  ~468 mA and look electrically alike.
- **Model comparison:** balanced accuracy of each member and of the ensemble
  on the same test set.
- **Per attack-group F1:** precision, recall, F1 and support (number of test
  rows) per group.

### Score your own telemetry

Upload a CSV with `time, shunt_voltage, bus_voltage_V, current_mA, power_mW`
(other columns optional). The server builds the same rolling features, scores
every row, and returns: rows scored, share flagged as attack, accuracy and
balanced accuracy (only if a `Label` column is present), a P(attack) trace, and
predicted group counts. Note it uses the currently trained models; with a
random split they have already seen most rows of the original dataset, so
uploading parts of it will look unrealistically good.

### Dataset card

Row counts, date range and class counts, read from the loaded CSV.

---

## 11. Code walkthrough

### `app/pipeline.py`

| Piece | What it does |
|---|---|
| `SIGNALS`, `WINDOW=20`, `METADATA` | Input columns, window size, columns never used as features. |
| `engineer(df)` | Parses time, sorts chronologically (stable), adds `_delta/_mean/_std/_skew/_crest` per signal, drops the first incomplete-window rows. |
| `make_ensemble()` | The XGB + LGBM + RF `VotingClassifier(voting="soft")`. |
| `Engine.load()` | Reads the CSV once and caches the feature table. |
| `Engine.train(strategy)` | 1) builds the train mask for `block`, `chrono` or `random`; 2) fits the Isolation Forest on training signals and adds its score; 3) fits the ensemble (binary) and multiclass XGBoost; 4) predicts on test rows, computes every metric, confusion matrices, per-model scores; 5) computes global TreeSHAP on 2,000 test rows; 6) attaches predictions to the test table and finds contiguous attack episodes. |
| `Engine.train_async()` | Runs training in a background thread and updates `status` so the UI can show progress. |
| `Engine.stream(cursor, n)` | Returns the next `n` test rows (with predictions) and a fresh TreeSHAP explanation of the last one. |
| `Engine.predict_csv(bytes)` | Scores an uploaded CSV with the trained models. |

Detail worth knowing: the binary `LabelEncoder` sorts classes alphabetically,
so `attack = 0`, `benign = 1`. The code looks up `attack_idx` instead of
assuming 1, and flips the SHAP sign so "+" always means "toward attack".

### `app/server.py` (FastAPI)

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves the page |
| `GET /api/status` | Training state, step, progress, trained strategy |
| `POST /api/train` | `{"strategy": "block" \| "chrono" \| "random"}`, starts retraining |
| `GET /api/metrics` | Held-out metrics, confusion matrices, global SHAP |
| `GET /api/segments` | Attack episodes for "Jump to" |
| `GET /api/stream?cursor=&n=` | Next rows of the replay |
| `POST /api/predict` | Score an uploaded CSV |
| `GET /api/dataset` | Dataset summary |

On startup it automatically trains with the block split.

### `app/static/main.js`

Polls `/api/status`; when models become ready it loads metrics and segments
and renders the evaluation section. The play loop calls `/api/stream` every
~450 ms, keeps the last 240 rows in a buffer, updates running statistics,
redraws the canvas charts, and passes each row to the 3D scene. anime.js
animates the gauge, numbers, confusion cells and progress bar.

### `app/static/charger.js`

Builds the three.js scene: a clay-material charger (rounded boxes, canopy,
screen drawn on a canvas texture, LED bar, holster, plug, cable as a tube along
a curve), a ground disc, halo rings and particles. `update(row)` sets the
target attack probability, the particle speed (from current) and redraws the
screen; the render loop smoothly blends colours from purple to yellow. If
`app/static/models/ev_charger.glb` exists, it is loaded in place of the
built-in body.

### Notebooks

| Notebook | Split | Features | Models | Purpose |
|---|---|---|---|---|
| 01 | Random rows | rolling mean/std/max/min | XGBoost | Show the leakage trap (~99.95% accuracy) |
| 02 | Chronological | full transient set + Isolation Forest | Ensemble + multiclass XGB + SHAP | The full pipeline under a strict future test |
| 03 | Chronological vs. 1-min blocks | full transient set | Ensemble + multiclass XGB | Direct comparison of the two honest splits |

The app's `pipeline.py` is a cleaned-up, runnable version of notebooks 02/03.

---

## 12. Benchmark against published work

Full method, literature review and tables:
[`benchmarks/evdetect/README.md`](../benchmarks/evdetect/README.md). Re-run
with `python benchmarks/evdetect/run_benchmark.py`.

**Benchmark chosen:** **EVDetect** (Lazaridis et al., *Scientific Reports*,
2026). It is the only peer-reviewed study on this exact file that publishes
code, its exact test split, tuned hyperparameters and confidence intervals.
Other power-data papers either publish no code, use different label sets, or
(the original dataset paper) report no power-data baseline.

**How it was run:**

- EVDetect's protocol reproduced exactly (stratified random row split, seed
  42, same test rows). Re-training their tuned RF and XGBoost here matches
  their published numbers within 0.002, which confirms a faithful reproduction.
- A second, leakage-aware protocol (1-minute blocks, block bootstrap) added,
  because a random row split leaks on time-series data (section 7).
- Four scenarios: binary and 5-class (Benign, DoS, Recon, Backdoor,
  Cryptojacking), on the full dataset and on charging-state rows.

**Results (macro-F1, leakage-aware block split):**

| Scenario | Best EVDetect | Ours, raw signals | Ours, full pipeline |
|---|---|---|---|
| Binary, full | 0.876 | 0.852 | **0.953** |
| Binary, charging | 0.802 | 0.791 | **0.939** |
| 5-class, full | 0.652 | 0.624 | **0.819** |
| 5-class, charging | 0.837 | 0.832 | **0.954** |

**What it means:**

- The project's pipeline wins in every scenario under both protocols, with
  non-overlapping 95% confidence intervals.
- "Ours, raw signals" (the same classifiers without the temporal features) is
  level with EVDetect. So the improvement comes from the **rolling-window
  features and the anomaly score**, not from using a fancier classifier.
- The advantage barely shrinks when moving to the leakage-aware split, so it is
  not a leakage artefact. Cross-session generalisation (chronological split)
  remains unsolved for every approach.

In the web app, the **Benchmark vs. EVDetect** section shows these results:
one card per scenario, a bar per model (macro-F1, axis from 0.5 to 1.0),
black whiskers for the 95% confidence interval, the full pipeline in yellow,
and a switch between the two protocols. The results come from
`benchmarks/evdetect/results/benchmark_results.json`; the app does not retrain
the benchmark live (it takes ~2 minutes).

---

## 13. Limitations and open questions

- **Generalisation to new sessions is weak** (chronological split). The block
  result says the models recognise *known* behaviour well, not that they would
  catch an attack recorded on another day or another charger.
- **One device, one lab.** All data comes from a single Raspberry Pi charger.
- **Quiet attacks** (tcp-flood, scans) overlap with benign idle; DoS vs. recon
  is the weakest distinction.
- **Minute-level timestamps** prevent precise second-by-second alignment.
- **Block split still leaks a little** (same session on both sides).
- Possible next steps: session-level hold-out, purged/gapped splits, sequence
  models (1D-CNN, LSTM, Transformers) on raw windows, frequency-domain features
  (FFT), domain adaptation, threshold calibration per deployment.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| EVSE | Electric Vehicle Supply Equipment, i.e. a charger |
| OCPP | Open Charge Point Protocol, used between chargers and the operator's backend |
| DoS | Denial of Service: overwhelming the device with traffic |
| Recon | Reconnaissance: scanning to find weaknesses |
| Backdoor | Malware giving an attacker remote control |
| Cryptojacking | Secretly using the device to mine cryptocurrency |
| Supervised learning | Learning from labelled examples |
| Unsupervised learning | Finding structure without labels (e.g. Isolation Forest, clustering) |
| Classification | Predicting a category |
| Clustering | Grouping similar points without labels (not used in this project) |
| Feature | An input column to the model |
| Leakage | Test information reaching the model during training |
| Autocorrelation | Neighbouring time points being similar |
| Out-of-distribution (OOD) | Test data that differs systematically from training data |
| Ensemble | Combination of several models |
| Bagging / Boosting | Averaging independent models / sequentially correcting errors |
| Log-odds | `log(p / (1-p))`; the scale of XGBoost's raw output and of SHAP values |
| Softmax | Turns K scores into K probabilities summing to 1 |
| Bootstrap | Resampling with replacement to estimate uncertainty |
