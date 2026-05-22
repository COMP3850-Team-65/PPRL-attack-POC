from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent

DATA_DIR = _BASE / "data" / "external" / "BaselineDataSplits"
EMB_DIR = _BASE / "data" / "external" / "clinicalbert"
MODEL_DIR = _BASE / "outputs" / "models"
RESULT_DIR = _BASE / "outputs" / "results"
FIG_DIR = _BASE / "outputs" / "figures"

N_SHADOWS = 10
RANDOM_STATE = 42

SA_EPOCHS = 200
SA_BATCH_SIZE = 64
CLF_EPOCHS = 100
CLF_BATCH_SIZE = 64
MARGIN = 0.5
ENCODER_OUTPUT_DIM = 128
