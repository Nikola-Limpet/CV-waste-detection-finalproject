"""Training-only data augmentation pipeline using Keras preprocessing layers."""

import tensorflow as tf


def get_augmentation_layer():
    """Return a Sequential augmentation model for training data.

    Expects input images normalized to [0, 1].
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.055),   # ~±20 degrees
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augmentation")
