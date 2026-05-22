import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def build_encoder(input_dim: int, output_dim: int = 128) -> tuple[Model, Model]:
    encoder_input = Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(256, name="encoder_hidden")(encoder_input)
    x = layers.LeakyReLU(alpha=0.01)(x)
    encoder_output = layers.Dense(output_dim, name="encoder_output")(x)
    encoder = Model(encoder_input, encoder_output, name="contrastive_encoder")

    input1 = Input(shape=(input_dim,), name="input_1")
    input2 = Input(shape=(input_dim,), name="input_2")
    encoded1 = encoder(input1)
    encoded2 = encoder(input2)
    merged = layers.Concatenate(name="encoded_pair")([encoded1, encoded2])
    model = Model(inputs=[input1, input2], outputs=merged, name="siamese_encoder")
    return model, encoder


def contrastive_loss(margin: float = 1.0):
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        emb_dim = tf.shape(y_pred)[1] // 2
        encoded1 = y_pred[:, :emb_dim]
        encoded2 = y_pred[:, emb_dim:]
        distances = tf.norm(encoded1 - encoded2, axis=1)
        y_true = tf.cast(y_true, tf.float32)
        return tf.reduce_mean(
            y_true * tf.square(distances)
            + (1 - y_true) * tf.square(tf.maximum(margin - distances, 0))
        )

    return loss_fn


def build_classifier(input_dim: int) -> Model:
    input_diff = Input(shape=(input_dim,), name="diff_input")
    x = layers.Dense(128, activation="relu", name="clf_hidden_1")(input_diff)
    x = layers.Dense(64, activation="relu", name="clf_hidden_2")(x)
    output = layers.Dense(1, activation="sigmoid", name="match_prob")(x)
    clf = Model(inputs=input_diff, outputs=output, name="match_classifier")
    clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return clf
