"""Training utilities: callbacks, compile-and-fit, fine-tuning."""

import os

import tensorflow as tf
from tensorflow import keras

from src.models import unfreeze_layers

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_callbacks(model_name: str) -> list:
    """Standard callback set: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{model_name}.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
    ]


def train_model(model, train_ds, val_ds, epochs, class_weights,
                callbacks, lr=1e-3):
    """Compile with Adam + SparseCategoricalCrossentropy and fit.

    Returns the Keras History object.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    return history


def fine_tune_model(model, base_model, train_ds, val_ds, epochs,
                    class_weights, callbacks, num_layers=30, lr=1e-5):
    """Unfreeze last layers of backbone, recompile at lower lr, and continue training."""
    unfreeze_layers(model, base_model, num_layers=num_layers)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    return history
