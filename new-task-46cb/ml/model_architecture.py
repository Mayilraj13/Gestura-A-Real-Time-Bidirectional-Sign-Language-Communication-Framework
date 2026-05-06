"""
GRU + Self-Attention model for sign language recognition.

Why GRU + Attention over plain LSTM:
  - GRU is faster (fewer parameters, no cell state)
  - Bidirectional GRU captures forward and backward motion context
  - Self-Attention assigns importance weights to each frame
    so the model focuses on the most informative gesture frames
  - Combined: higher accuracy, lower latency, better GPU utilization
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class SelfAttention(layers.Layer):
    """Scaled dot-product self-attention over time steps."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units          # must store for get_config serialization
        self.scale = float(units) ** 0.5
        self.W_q = layers.Dense(units)
        self.W_k = layers.Dense(units)
        self.W_v = layers.Dense(units)

    def call(self, x, training=False):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, V)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model(num_classes: int, sequence_length: int = 20, feature_dim: int = 63) -> Model:
    """
    Build GRU + Self-Attention classifier.

    Args:
        num_classes:     number of sign classes
        sequence_length: frames per sequence (20)
        feature_dim:     keypoints per frame (21 × 3 = 63)

    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="keypoints")

    x = layers.Masking(mask_value=0.0)(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(
        layers.GRU(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bigru_1",
    )(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bigru_2",
    )(x)
    x = layers.Dropout(0.3)(x)

    x = SelfAttention(units=256, name="self_attention")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ETD_GRU_Attention")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")],
    )

    return model


def build_model_with_mixed_precision(num_classes: int, **kwargs) -> Model:
    """Build model with mixed-precision (float16) for RTX 2050 speedup."""
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = build_model(num_classes, **kwargs)
    return model
