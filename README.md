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
