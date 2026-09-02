# Results

## Summary

This project's central finding is a **validation-methodology warning**: the
naïve evaluation approach produces near-perfect scores that are not
representative of real-world generalization, while leakage-aware temporal
validation yields honest — and substantially lower — performance.

## Binary Detection (attack vs. benign)

| Validation Strategy | Balanced Accuracy | Attack Precision | Attack Recall | Attack F1 | Accuracy | Notes |
|---------------------|-------------------|------------------|---------------|-----------|----------|-------|
| Naïve random split (01) | ~0.999 | ~1.00 | ~1.00 | ~1.00 | **0.99948** | Leakage-inflated; not transferable |
| Chronological, OOD (02 / 03-S1) | **0.72** | 1.00 | 0.72 | 0.84 | — | Test set = attack-only future window; PR-AUC 0.50 |
| Stratified block (03-S2) | **0.95** | 0.99 | 0.99 | 0.99 | **0.98** | benign: prec 0.96 / rec 0.91 / f1 0.93 |

> **Interpretation.** The ~99.95% accuracy from the random split is an
> artifact of temporal leakage: temporally adjacent samples in train and test
> make the task artificially easy. Under a leakage-aware stratified-block
> split, detection remains strong (**bal. acc. 0.95**), but under the strict
> chronological out-of-distribution horizon, generalization drops
> substantially (**bal. acc. ~0.71–0.72**), driven largely by the model
> struggling to generalize attack patterns to unseen future conditions.

## Multiclass Attack-Group Identification

| Validation Strategy | Balanced Accuracy / Accuracy | Macro-F1 | Behaviour |
|---------------------|------------------------------|----------|-----------|
| Chronological, OOD (02 / 03-S1) | bal. acc. **0.08** | — | Collapses: only `host-attack` predicted |
| Stratified block (03-S2) | accuracy **0.80** | **0.81** | DoS 0.61 · host-attack 0.95 · none 0.93 · recon 0.74 |

> **Interpretation.** Multiclass attack *identification* is the hardest
> problem. On the strict chronological (out-of-distribution) test window the
> classifier degenerates to predicting a single class (balanced accuracy
> ~0.08), showing that finer-grained attack-type generalization to unseen
> temporal conditions is not yet achievable with this feature set. When both
> classes and all attack types are represented (stratified block), the model
> achieves ~0.80 accuracy / 0.81 macro-F1.

## Key Takeaways

1. **Power-consumption telemetry alone is a viable binary intrusion signal.**
   Balanced accuracy of 0.95 (leakage-aware) supports trainability of the
   detector from electrical signals only — no network-packet data required.
2. **Temporal leakage is the dominant risk** in this and similar power
   datasets; a random split is misleading (0.999 vs. 0.72–0.95 honest).
3. **Out-of-distribution (future-horizon) generalization is weak**, especially
   for multiclass attack-type identification (bal. acc. ~0.08), which remains
   open research.

## Figures

Generated plots (confusion matrices, ROC/PR curves, SHAP summaries, feature
importance) are produced by the notebooks and should be saved under
[`figures/`](../figures/). Formal report artifacts live under
[`reports/`](../reports/).
