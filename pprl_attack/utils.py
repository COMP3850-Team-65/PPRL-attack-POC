import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


def set_seeds(seed: int = 42) -> None:
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def verify_file_exists(path: str | Path) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stratified_bootstrap_sample(
    df: pd.DataFrame, n_samples: int, random_state: int = 42
) -> list[np.ndarray]:
    samples = []
    for i in range(n_samples):
        rng = np.random.RandomState(random_state + i)
        idx = []
        for label, group in df.groupby("label"):
            n = len(group)
            boot_idx = group.index[rng.choice(n, size=n, replace=True)]
            idx.append(boot_idx)
        combined = np.concatenate(idx)
        rng.shuffle(combined)
        samples.append(combined)
    return samples
