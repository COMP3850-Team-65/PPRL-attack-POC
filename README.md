# pprl-attack-poc

Proof-of-concept Membership Inference Attack (MIA) against a Privacy-Preserving Record Linkage (PPRL) model.

## What this is

A linkage model takes pairs of clinical notes and predicts whether they describe the same patient. This is a useful tool for hospitals but raises a privacy question: can an attacker tell whether a specific record was used to train the model? This project demonstrates a working attack that answers that question with above-chance accuracy.

The attack follows the standard Shokri et al. shadow-model approach. We train a target model (the victim), train an architecturally-identical shadow model on disjoint data (the attacker's stand-in), use the shadow's known membership labels to teach an attack classifier what "trained on" looks like, and apply that classifier to the target.

## Setup

### NixOS (recommended)

```bash
nix-shell
```

`shell.nix` creates a Python 3.11 venv at `.venv/`, syncs `requirements.txt`, and exports `LD_LIBRARY_PATH` so compiled extensions (zmq, TensorFlow) load correctly.

VSCode needs the same environment to launch Jupyter cells. Either launch VSCodium from inside `nix-shell` or set up `direnv` with `nix-direnv` so the environment loads automatically when entering the directory.

### Other systems

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Attack

The attack is split across 4 notebooks that must be run in order:

- `00_baseline_snn_mlp.ipynb` - trains the target SNN+MLP on the full dataset
- `01_shadow_snn_mlp.ipynb` - trains the shadow model (same architecture, disjoint data)
- `02_mia_feature_extraction.ipynb` - scores all pairs through both models and extracts `prob` + `loss` features
- `03a_mia_attack.ipynb` - trains the attack classifier and evaluates its results

`00` and `01` are functionally identical, the only difference is which data split they train on. The shadow model acts as a proxy for the target: the attacker trains their classifier on shadow features (where membership labels are known) and evaluates it against the target (where they aren't).

A fifth notebook `03b_mia_attack_encoder_distance.ipynb` is planned for Stage 2, adding encoder distance as a third attack feature to hopefully improve results.
