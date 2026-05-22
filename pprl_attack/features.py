import numpy as np
import pandas as pd
from tensorflow.keras.models import Model


def score_pairs(
    encoder: Model, clf: Model, x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    enc1 = encoder.predict(x1, verbose=0)
    enc2 = encoder.predict(x2, verbose=0)
    diff = np.abs(enc1 - enc2)
    return clf.predict(diff, verbose=0).flatten()


def per_pair_loss(
    probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-7
) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    p = np.clip(probs, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def correctness_confidence(probs: np.ndarray) -> np.ndarray:
    return np.maximum(probs, 1 - probs)


def per_pair_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    p = np.clip(probs, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def build_feature_frame(
    probs: np.ndarray,
    y_true: np.ndarray,
    member_label: int,
    source_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prob": probs,
            "loss": per_pair_loss(probs, y_true),
            "correctness_confidence": correctness_confidence(probs),
            "entropy": per_pair_entropy(probs),
            "prob_correct": np.where(y_true == 1, probs, 1 - probs),
            "y_true": y_true,
            "member": member_label,
            "source": source_name,
        }
    )
