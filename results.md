# Results

## Phase 1: Legacy (autoencoder, single shadow)

Initial approach using an autoencoder-based architecture with a single shadow model. Target and shadow used different encoder architectures, which limited the transfer of membership signal.

**Models**

| Model | Train acc | Test acc | Gap |
|---|---|---|---|
| Target | 0.789 | 0.753 | +0.036 |
| Shadow | 0.736 | 0.718 | +0.018 |

**Attack**

| Metric | Value |
|---|---|
| Accuracy | 0.673 |
| AUC | 0.522 |
| TPR @ FPR=1% | 0.010 |
| TPR @ FPR=10% | 0.104 |

![Phase 1 ROC](outputs/figures/mia_roc_phase1.png)

---

## What changed (Phase 1 -> Phase 2)

- Unified target/shadow encoder architecture (identical Siamese network)
- Switched from autoencoder to contrastive loss (margin=0.5)
- Multi-shadow bootstrap sampling (10 shadows, N=1 with replacement)
- Disjoint per-shadow test folds via StratifiedKFold
- Extended attack to 5 features (added `prob_correct`, `entropy`)
- Hyperparameter-tuned attack classifier (GridSearchCV + balanced class weights)

---

## Phase 2: Siamese encoder + 10-shadow bootstrap

Target and shadow models share an identical architecture trained on pre-computed ClinicalBERT embeddings with disjoint per-shadow test folds.

**Pipeline config**

| Parameter | Target | Shadows |
|---|---|---|
| Encoder | `Input(768)->Dense(256)->LeakyReLU->Dense(128)` | Identical |
| Contrastive margin | 0.5 | 0.5 |
| Encoder epochs | 200 | 200 |
| Classifier | `Dense(128)->Dense(64)->Dense(1)` on \|enc1–enc2\| | Identical |
| Classifier epochs | 100 | 100 |
| Batch size | 64 | 64 |
| Shadow sampling | — | Stratified bootstrap, N=10 |

**Linkage model performance**

| Model | Test accuracy |
|---|---|
| target | 0.747 |
| shadow_0–shadow_9 | 0.665–0.755 |
| **Shadow mean** | **0.700** |

**Attack performance**

| Metric | Value |
|---|---|
| AUC | 0.637 |
| 95% CI | [0.630, 0.645] |
| TPR @ FPR=1% | 0.017 |
| TPR @ FPR=10% | 0.171 |

AUC of 0.637 is 13.7 points above random baseline with tight confidence intervals. Within-model MIA on the target stays at 0.515, suggesting the shadow-model transfer extracts a membership signal the target doesn't exhibit internally.

![Phase 2 ROC](outputs/figures/mia_roc_phase2.png)
