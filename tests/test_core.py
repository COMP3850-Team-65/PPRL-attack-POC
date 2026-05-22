import numpy as np
import pandas as pd
import tensorflow as tf
from pprl_attack.models import build_encoder, contrastive_loss, build_classifier
from pprl_attack.features import (
    score_pairs, per_pair_loss, correctness_confidence,
    per_pair_entropy, build_feature_frame,
)


class TestContrastiveLoss:
    def test_perfect_match_loss_is_zero(self):
        loss_fn = contrastive_loss(margin=1.0)
        y_true = tf.constant([[1.0]])
        y_pred = tf.constant([[5.0, 5.0]])
        loss = loss_fn(y_true, y_pred).numpy()
        assert loss == 0.0

    def test_non_match_beyond_margin(self):
        loss_fn = contrastive_loss(margin=1.0)
        y_true = tf.constant([[0.0]])
        y_pred = tf.constant([[0.0, 10.0]])
        loss = loss_fn(y_true, y_pred).numpy()
        assert loss == 0.0

    def test_non_match_within_margin(self):
        loss_fn = contrastive_loss(margin=1.0)
        y_true = tf.constant([[0.0]])
        y_pred = tf.constant([[0.0, 0.3]])
        loss = loss_fn(y_true, y_pred).numpy()
        assert loss > 0.0


class TestPerPairLoss:
    def test_perfect_prediction(self):
        probs = np.array([1.0, 0.0])
        y_true = np.array([1, 0])
        loss = per_pair_loss(probs, y_true)
        assert np.allclose(loss, [0.0, 0.0], atol=1e-6)

    def test_clipping_prevents_log_zero(self):
        probs = np.array([0.0])
        y_true = np.array([0])
        loss = per_pair_loss(probs, y_true)
        assert np.isfinite(loss[0])

    def test_wrong_prediction_high_loss(self):
        probs = np.array([0.0])
        y_true = np.array([1])
        loss = per_pair_loss(probs, y_true)
        assert loss[0] > 10.0


class TestCorrectnessConfidence:
    def test_high_confidence(self):
        assert correctness_confidence(np.array([0.95])) == 0.95
        assert correctness_confidence(np.array([0.1])) == 0.9

    def test_random_guess(self):
        assert correctness_confidence(np.array([0.5])) == 0.5


class TestPerPairEntropy:
    def test_certain_prediction(self):
        assert np.isclose(per_pair_entropy(np.array([1.0]))[0], 0.0)

    def test_max_entropy_at_05(self):
        h = per_pair_entropy(np.array([0.5]))[0]
        assert np.isclose(h, 1.0, atol=1e-6)

    def test_entropy_symmetric(self):
        h1 = per_pair_entropy(np.array([0.3]))[0]
        h2 = per_pair_entropy(np.array([0.7]))[0]
        assert np.isclose(h1, h2)


class TestBuildFeatureFrame:
    def test_all_columns_present(self):
        probs = np.array([0.9, 0.2])
        y_true = np.array([1, 0])
        df = build_feature_frame(probs, y_true, member_label=1, source_name="test")
        expected_cols = {
            "prob", "loss", "correctness_confidence", "entropy",
            "prob_correct", "y_true", "member", "source",
        }
        assert set(df.columns) == expected_cols

    def test_member_label_assigned(self):
        df = build_feature_frame(
            np.array([0.5]), np.array([1]), member_label=1, source_name="x"
        )
        assert df["member"].iloc[0] == 1

    def test_source_name_assigned(self):
        df = build_feature_frame(
            np.array([0.5]), np.array([1]), member_label=0, source_name="target_train"
        )
        assert df["source"].iloc[0] == "target_train"

    def test_correctness_confidence_matches_function(self):
        probs = np.array([0.9, 0.2, 0.5])
        y_true = np.array([1, 0, 1])
        df = build_feature_frame(probs, y_true, member_label=0, source_name="test")
        assert np.allclose(df["correctness_confidence"], correctness_confidence(probs))


class TestBuildEncoder:
    def test_output_shapes(self):
        sa_model, encoder = build_encoder(input_dim=768, output_dim=128)
        assert encoder.input_shape == (None, 768)
        assert encoder.output_shape == (None, 128)
        assert len(sa_model.inputs) == 2
        assert sa_model.outputs[0].shape[1] == 256


class TestBuildClassifier:
    def test_output_shape(self):
        clf = build_classifier(input_dim=128)
        x = np.random.randn(4, 128).astype(np.float32)
        out = clf.predict(x, verbose=0)
        assert out.shape == (4, 1)


class TestScorePairs:
    def test_output_shape(self):
        input_dim = 4
        sa_model, encoder = build_encoder(input_dim, output_dim=2)
        clf = build_classifier(2)
        x1 = np.random.randn(10, input_dim).astype(np.float32)
        x2 = np.random.randn(10, input_dim).astype(np.float32)
        probs = score_pairs(encoder, clf, x1, x2)
        assert probs.shape == (10,)
