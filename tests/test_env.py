def test_core_imports():
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from transformers import BertTokenizer

    assert np.__version__.startswith("1.")
    assert tf.__version__.startswith("2.15")
