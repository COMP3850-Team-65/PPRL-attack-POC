# pprl-attack-poc

Proof-of-concept Membership Inference Attack (MIA) against a Privacy-Preserving Record Linkage (PPRL) model.

## What this is

A linkage model takes pairs of clinical notes and predicts whether they describe the same patient. This is a useful tool for hospitals but raises a privacy question: can an attacker tell whether a specific record was used to train the model? This project demonstrates a working attack that answers that question with above-chance accuracy.

The attack follows the standard Shokri et al. shadow-model approach. We train a target model (the victim), train architecturally-identical shadow models on disjoint data (the attacker's stand-in), use the shadow's known membership labels to teach an attack classifier what "trained on" looks like, and apply that classifier to the target.

## Repository structure

```
data/
  external/
    BaselineDataSplits/   # target_train.csv, target_test.csv, shadow_train.csv, shadow_test.csv
    clinicalbert/         # pre-computed ClinicalBERT embeddings (.npy)
notebooks/
  00_train_models.ipynb   # Step 1: train target + 10 shadow models
  01_attack.ipynb         # Step 2: extract features, train attack classifier
outputs/
  models/                 # saved .h5 models and .npy training subsets
  results/                # output CSVs (mia_features_*.csv, mia_attack_metrics.csv)
  figures/                # ROC curve plots
pprl_attack/              # supporting Python package (config, models, features, utils, run_logger)
scripts/
  check_patient_disjointness.py  # post-run verification script
requirements.txt
shell.nix                 # NixOS dev shell (optional)
results.md
```

## Data preparation

Synthetic clinical note pairs were provided by MQ SOC. The dataset is not included in this repository.

The pipeline expects pre-computed ClinicalBERT embeddings in `data/external/clinicalbert/` as `.npy` files. The embedding generation steps and the train/test split procedure (the four CSV files in `data/external/BaselineDataSplits/`) were performed by MQ SOC and are not documented in this repository.

## Setup

### Requirements

Python 3.11 is required by TensorFlow 2.15.1. All dependencies are listed in `requirements.txt`. Only `tensorflow` is hard-pinned (`==2.15.1`); the remaining packages use lower-bound constraints. For exact reproducibility, pin all dependencies by running:

```bash
pip freeze > requirements-pinned.txt
```

and use `requirements-pinned.txt` for any fresh environment.

### NixOS (recommended)

```bash
nix-shell
```

`shell.nix` creates a Python 3.11 venv at `.venv/`, syncs `requirements.txt`, and exports `LD_LIBRARY_PATH` so compiled extensions (zmq, TensorFlow) load correctly. It also sets `PYTHONPATH=$PWD` automatically via `shellHook`, which is required for the `pprl_attack` package to resolve.

VSCode needs the same environment to launch Jupyter cells. Either launch VSCodium from inside `nix-shell` or set up `direnv` with `nix-direnv` so the environment loads automatically when entering the directory.

### Other systems

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Launching Jupyter

```bash
jupyter notebook
```

Open `notebooks/` in the browser and run the notebooks in order. Notebooks must be run from the repository root so that relative paths in `pprl_attack/config.py` resolve correctly.

## Configuration

All pipeline parameters are set in `pprl_attack/config.py`. The table below lists each parameter so you can modify a run without reading source code.

| Parameter | Value | Description |
|---|---|---|
| N_SHADOWS | 10 | Number of shadow models |
| RANDOM_STATE | 42 | Global random seed |
| SA_EPOCHS | 200 | Encoder (contrastive) training epochs |
| SA_BATCH_SIZE | 64 | Encoder batch size |
| CLF_EPOCHS | 100 | MLP classifier training epochs |
| CLF_BATCH_SIZE | 64 | Classifier batch size |
| MARGIN | 0.5 | Contrastive loss margin |
| ENCODER_OUTPUT_DIM | 128 | Encoder output embedding dimension |

## Running the pipeline

The pipeline is split across two notebooks that must be run in order.

### Step 1: `00_train_models.ipynb`

Trains the target encoder, a contrastive Siamese network (`Input(768) -> Dense(256) -> LeakyReLU -> Dense(128)`), and its MLP classifier head. Then trains 10 shadow models with identical architecture on stratified bootstrap samples of the shadow data, using disjoint per-shadow test folds via `StratifiedKFold`.

**Inputs:**
- `data/external/BaselineDataSplits/target_train.csv`
- `data/external/BaselineDataSplits/target_test.csv`
- `data/external/BaselineDataSplits/shadow_train.csv`
- `data/external/BaselineDataSplits/shadow_test.csv`
- `data/external/clinicalbert/*.npy` (all 12 embedding files)

**Outputs** (written to `outputs/models/`):
- `target_encoder.h5`, `target_clf.h5`
- `shadow_encoder_{i}.h5`, `shadow_clf_{i}.h5` (i = 0..9)
- `shadow_x1_train_{i}.npy`, `shadow_x2_train_{i}.npy`, `shadow_y_train_{i}.npy`
- `test_fold_indices.npy`
- `shadow_accuracy.csv`

**Expected runtime:** approximately 30 minutes on a CPU-only machine (measured at 1,769 seconds for the locked-in run).

### Step 2: `01_attack.ipynb`

Loads the saved models and training subsets from Step 1. For each shadow model, it scores the shadow's own training pairs (member = 1) and its unique test fold (member = 0), extracting five attack features per pair:

- `prob`: raw classifier output probability
- `loss`: per-pair binary cross-entropy loss
- `correctness_confidence`: max(prob, 1 - prob)
- `entropy`: per-pair binary Shannon entropy
- `prob_correct`: probability assigned to the true label

The shadow features are aggregated into a single training table for the attack classifier (logistic regression with `GridSearchCV` over `C in [0.01, 0.1, 1, 10, 100]`, 5-fold CV, scoring `roc_auc`, balanced class weights). The same features are extracted from the target model for evaluation. A 1000-sample bootstrap confidence interval is computed on the target AUC.

**Outputs:**
- `outputs/results/mia_features_shadow.csv`
- `outputs/results/mia_features_target.csv`
- `outputs/results/mia_attack_metrics.csv`
- `outputs/figures/mia_roc_phase2.png`

### Post-run verification

After both notebooks complete, verify patient-level disjointness between target and shadow splits:

```bash
python scripts/check_patient_disjointness.py
```

## Results

### Linkage model performance

| Model | Train acc | Test acc | Gap |
|---|---|---|---|
| Target | 0.885 | 0.705 | +0.180 |
| Shadow mean | 0.960 | 0.703 | +0.257 |

The target model achieves 0.885 training accuracy and 0.705 test accuracy, a generalisation gap of 0.180. Shadow models show a wider gap (mean 0.257) due to their bootstrap training procedure.

### Attack performance

| Metric | Value |
|---|---|
| Shadow-transfer AUC | 0.628 |
| 95% CI | [0.620, 0.635] |
| TPR @ 1% FPR | 0.014 |
| TPR @ 10% FPR | 0.150 |
| Within-model oracle AUC | 0.631 |

The shadow-transfer attack achieves an AUC of 0.628, closely matching the within-model oracle of 0.631. This shows that the shadow-model methodology recovers the full membership signal available under black-box constraints. The oracle AUC of 0.631 confirms that the target model's moderate generalisation gap (+0.180) places a ceiling on any membership inference attack against this model.

### Interpreting outputs

`mia_attack_metrics.csv` contains the full set of logged metrics per run. `mia_roc_phase2.png` shows the ROC curve for the shadow-transfer attack against the random baseline. A re-run should reproduce AUC within the reported 95% confidence interval; exact values may differ slightly due to non-determinism in TensorFlow (though `TF_DETERMINISTIC_OPS=1` is set automatically by `pprl_attack/utils.py set_seeds()`).

## Key takeaways

The attack demonstrates that a PPRL model trained on clinical note pairs leaks membership information under black-box constraints. The shadow-model transfer achieves an AUC of 0.628, closely matching the oracle within-model AUC of 0.631, which shows that the shadow-model methodology successfully recovers the available membership signal without direct access to the target model's training data. The within-model baseline confirms that the target model's small generalisation gap limits the ceiling of any membership inference attack, so the primary risk factor for a production PPRL system is overfitting rather than architectural choice. MQ SOC can use these results to benchmark the privacy exposure of PPRL deployments and to motivate differential privacy or other regularisation controls.

## Limitations

This attack is a proof of concept on synthetic data only. Findings may not transfer directly to real clinical records without re-evaluation on non-synthetic data. The black-box constraint assumed here (score-only access) is realistic but more permissive threat models (white-box, gradient access) may yield higher attack AUC.
